import contextlib

import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm
from torch import autograd

from models.base import get_embedder


def interpolation(features: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor):
    fg = torch.sum(features[indices] * weights.unsqueeze(-1), dim=-2)
    return fg


class NeuMesh(nn.Module):
    def __init__(
        self,
        mesh_grid,
        D_density: int,
        D_color: int,
        W: int,
        geometry_dim: int,
        color_dim: int,
        multires_view: int,
        multires_d: int,
        multires_fg: int,
        multires_ft: int,
        enable_nablas_input: bool,
        input_view_dim=3,
        input_d_dim=1,
        ln_s=0.2996,
        speed_factor=1.0,
        learn_indicator_weight=True,
    ):
        super(NeuMesh, self).__init__()

        # initialize mesh grid
        self.mesh_grid = mesh_grid
        num_vertices = self.mesh_grid.get_number_of_vertices()

        # initialize scale factor
        self.ln_s = nn.Parameter(torch.Tensor([ln_s]), requires_grad=True)
        self.speed_factor = speed_factor

        # initialize geometry/texture features
        self.geometry_features = nn.Parameter(
            torch.randn(num_vertices, geometry_dim, dtype=torch.float32)
        )
        self.color_features = nn.Parameter(
            torch.randn(num_vertices, color_dim, dtype=torch.float32)
        )

        # initialize indicator vectors
        indicator_vector = self.mesh_grid.get_vertex_normal_torch().float().clone()
        self.indicator_vector = nn.Parameter(indicator_vector)
        self.learn_indicator_weight = learn_indicator_weight
        if self.learn_indicator_weight:
            self.indicator_weight_raw = nn.Parameter(
                torch.Tensor([-2]), requires_grad=True
            )

        # initialize embeddings of distance, directions, geometry/texture features
        self.embed_fn_d, input_ch_d = get_embedder(multires_d, input_dim=input_d_dim)
        self.embed_fn_view, input_ch_view = get_embedder(
            multires_view, input_dim=input_view_dim
        )
        self.embed_fn_fg, input_ch_fg = get_embedder(
            multires_fg, input_dim=geometry_dim
        )
        self.embed_fn_ft, input_ch_ft = get_embedder(multires_ft, input_dim=color_dim)

        # initialize geometry MLP
        input_ch_pts = input_ch_d + input_ch_fg
        self.softplus = nn.Softplus(beta=100)
        self.pts_linears = nn.Sequential(
            weight_norm(nn.Linear(input_ch_pts, W)),
            self.softplus,
            *[
                nn.Sequential(
                    weight_norm(nn.Linear(W, W)),
                    self.softplus,
                )
                for i in range(D_density - 1)
            ],
        )

        # initialize geometry MLP
        input_ch_color = input_ch_view + input_ch_ft + input_ch_d
        self.enable_nablas_input = enable_nablas_input
        if self.enable_nablas_input:
            input_ch_color += 3
        self.views_linears = nn.Sequential(
            nn.Linear(input_ch_color, W),
            nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(W, W), nn.ReLU(inplace=True))
                for i in range(D_color - 1)
            ],
        )
        self.density_linear = weight_norm(nn.Linear(W, 1))
        self.color_linear = nn.Sequential(nn.Linear(W, 3), nn.Sigmoid())

        # log the model information
        print(
            f"[Info] input_d_dim: {input_d_dim}, input_ch_d: {input_ch_d}, "
            + f"input_view_dim: {input_view_dim}, input_ch_view: {input_ch_view}, "
            + f"geometry_dim: {geometry_dim}, input_ch_fg: {input_ch_fg}, "
            + f"color_dim: {color_dim}, input_ch_ft: {input_ch_ft}, "
            + f"input_ch_pts: {input_ch_pts}, input_ch_color: {input_ch_color}\n"
        )

    def forward(
        self,
        xyz: torch.Tensor,
        view_dirs: torch.Tensor,
        need_nablas=True,
        nablas_only=False,
        return_ds=False,
    ):
        if need_nablas:
            xyz.requires_grad_(True)
        with (torch.enable_grad() if need_nablas else contextlib.nullcontext()):
            ds, indices, weights = self.compute_distance(xyz)
        out = self._forward(
            xyz,
            ds,
            view_dirs,
            indices,
            weights,
            need_nablas=need_nablas,
            nablas_only_for_eikonal=nablas_only,
        )

        if return_ds == True:
            out = out + (ds, indices, weights)

        return out

    def forward_density_only(self, xyz):
        ds, indices, weights = self.compute_distance(xyz)
        density, _, _ = self._forward_density(
            xyz, ds, self.geometry_features, indices, weights, need_nablas=False
        )
        return density

    def forward_with_nablas(self, xyz: torch.Tensor):
        xyz.requires_grad_(True)
        with torch.enable_grad():
            ds, indices, weights = self.compute_distance(xyz)
        density, nablas, _ = self._forward_density(
            xyz, ds, self.geometry_features, indices, weights, need_nablas=True
        )
        return density, nablas

    def forward_color(
        self,
        d: torch.Tensor,
        view_dirs: torch.Tensor,
        color_features: torch.Tensor,
        indices: torch.Tensor = None,
        weights: torch.Tensor = None,
        nabla=None,
    ):
        d_emb = self.embed_fn_d(d)
        return self._forward_color(
            d_emb, view_dirs, color_features, indices, weights, nabla
        )

    def forward_s(self):
        return torch.exp(self.ln_s * self.speed_factor)

    def forward_indicator_weight(self):
        return torch.sigmoid(self.indicator_weight_raw)

    def _forward(
        self,
        xyz: torch.Tensor,
        d: torch.Tensor,
        view_dirs: torch.Tensor,
        indices: torch.Tensor = None,
        weights: torch.Tensor = None,
        need_nablas: bool = False,
        nablas_only_for_eikonal: bool = False,
    ):

        density, nablas, d_emb = self._forward_density(
            xyz,
            d,
            self.geometry_features,
            indices,
            weights,
            need_nablas=need_nablas,
        )

        if nablas_only_for_eikonal:
            return density, nablas

        color = self._forward_color(
            d_emb, view_dirs, self.color_features, indices, weights, nablas
        )
        return density, color

    def _forward_density(
        self,
        xyz: torch.Tensor,
        d: torch.Tensor,
        geometry_features: torch.Tensor,
        indices: torch.Tensor = None,
        weights: torch.Tensor = None,
        need_nablas: bool = False,
    ):
        with (torch.enable_grad() if need_nablas else contextlib.nullcontext()):
            d_emb = self.embed_fn_d(d)
            fg = interpolation(geometry_features, indices, weights)
            fg_emb = self.embed_fn_fg(fg)
            h = self.pts_linears(torch.cat([d_emb, fg_emb], dim=-1))
            density = self.density_linear(h)

        if need_nablas == False:
            return density, torch.zeros_like(density), d_emb

        has_grad = torch.is_grad_enabled()
        if need_nablas:
            nabla = autograd.grad(
                density,
                xyz,
                torch.ones_like(density, device=xyz.device),
                create_graph=has_grad,
                retain_graph=has_grad,
                only_inputs=True,
            )[0]

        if not has_grad:
            nabla = nabla.detach()

        return density, nabla, d_emb

    def _forward_color(
        self,
        d_emb: torch.Tensor,
        view_dirs: torch.Tensor,
        color_features: torch.Tensor,
        indices: torch.Tensor = None,
        weights: torch.Tensor = None,
        nabla: torch.Tensor = None,
    ):
        view_dirs_emb = self.embed_fn_view(view_dirs)
        color_input = []
        if self.enable_nablas_input:
            color_input.append(nabla)
        color_input.append(d_emb)
        color_input.append(view_dirs_emb)
        ft = interpolation(color_features, indices, weights)
        ft_emb = self.embed_fn_ft(ft)
        color_input.append(ft_emb)

        hv = self.views_linears(torch.cat(color_input, dim=-1))  # [(h), dir_emb, ft]
        color = self.color_linear(hv)
        return color

    def compute_distance(self, xyz):
        ds, indices, weights = self.mesh_grid.compute_distance(
            xyz.view(-1, 3),
            indicator_vector=self.indicator_vector,
            indicator_weight=self.forward_indicator_weight()
            if self.learn_indicator_weight
            else 0.1,
        )
        ds = ds.reshape(*xyz.shape[:-1], -1)
        indices = indices.reshape(*xyz.shape[:-1], -1)
        weights = weights.reshape(*xyz.shape[:-1], -1)
        return ds, indices, weights
