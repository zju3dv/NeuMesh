import functools
from tqdm import tqdm
from typing import Optional
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import rend_util, train_util


def cdf_Phi_s(x, s):
    return torch.sigmoid(x * s)


def sdf_to_alpha(sdf: torch.Tensor, s):
    # [(B), N_rays, N_pts]
    cdf = cdf_Phi_s(sdf, s)
    # [(B), N_rays, N_pts-1]
    # TODO: check sanity.
    opacity_alpha = (cdf[..., :-1] - cdf[..., 1:]) / (cdf[..., :-1] + 1e-10)
    opacity_alpha = torch.clamp_min(opacity_alpha, 0)
    return cdf, opacity_alpha


def sdf_to_w(sdf: torch.Tensor, s):
    device = sdf.device
    # [(B), N_rays, N_pts-1]
    cdf, opacity_alpha = sdf_to_alpha(sdf, s)

    # [(B), N_rays, N_pts]
    shifted_transparency = torch.cat(
        [
            torch.ones([*opacity_alpha.shape[:-1], 1], device=device),
            1.0 - opacity_alpha + 1e-10,
        ],
        dim=-1,
    )

    # [(B), N_rays, N_pts-1]
    visibility_weights = (
        opacity_alpha * torch.cumprod(shifted_transparency, dim=-1)[..., :-1]
    )

    return cdf, opacity_alpha, visibility_weights


def alpha_to_w(alpha: torch.Tensor):
    device = alpha.device
    # [(B), N_rays, N_pts]
    shifted_transparency = torch.cat(
        [
            torch.ones([*alpha.shape[:-1], 1], device=device),
            1.0 - alpha + 1e-10,
        ],
        dim=-1,
    )

    # [(B), N_rays, N_pts-1]
    visibility_weights = alpha * torch.cumprod(shifted_transparency, dim=-1)[..., :-1]

    return visibility_weights


def compute_bounded_near_far(
    model,
    rays_o,
    rays_d,
    near,
    far,
    sample_grid: int = 256,
    distance_thresh: float = 0.1,
):
    near_orig = near.clone()
    far_orig = far.clone()
    # rays_o, rays_d: (1, N_rays, 3)
    # near, far: (1, N_rays, 1)
    _t = torch.linspace(0, 1, sample_grid, device=rays_o.device)
    # d_coarse: (1, N_rays, N_grid)
    d_coarse = near * (1 - _t) + far * _t
    # d_coarse: (1, N_rays, N_grid, 1)
    d_coarse = d_coarse.unsqueeze(-1)
    # pts_coarse: (1, N_rays, N_grid, 3)
    pts_coarse = rays_o.unsqueeze(-2) + d_coarse * rays_d.unsqueeze(-2)
    ds, _, _ = model.compute_distance(pts_coarse)
    mask = ds < distance_thresh

    near = d_coarse * mask.float() + (~mask).float() * 1e10
    near = near.min(dim=-2, keepdim=False)[0]
    near_mask = near > 1e5
    near[near_mask] = near_orig[near_mask]

    far = d_coarse * mask.float() - (~mask).float() * 1e10
    far = far.max(dim=-2, keepdim=False)[0]
    far_mask = far < -1e5
    far[far_mask] = far_orig[far_mask]
    # compensate too small near far
    too_close = (far - near) < 0.1
    far[too_close] += 0.05
    near[too_close] -= 0.05
    return near, far


def volume_render(
    rays_o,
    rays_d,
    model,
    obj_bounding_radius=1.0,
    batched=False,
    batched_info={},
    # render algorithm config
    calc_normal=False,
    use_view_dirs=True,
    rayschunk=65536,
    netchunk=1048576,
    white_bkgd=False,
    near_bypass: Optional[float] = None,
    far_bypass: Optional[float] = None,
    # render function config
    detailed_output=True,
    show_progress=False,
    # sampling related
    perturb=False,  # config whether do stratified sampling
    fixed_s_recp=1 / 64.0,
    N_samples=64,
    N_importance=64,
    # upsample related
    N_nograd_samples=2048,
    N_upsample_iters=4,
    samples_output=False,
    bounded_near_far=True,
    random_color_direction=False,
    **dummy_kwargs,  # just place holder
):
    """
    input:
        rays_o: [(B,) N_rays, 3]
        rays_d: [(B,) N_rays, 3] NOTE: not normalized. contains info about ratio of len(this ray)/len(principle ray)
    """
    device = rays_o.device
    if batched:
        DIM_BATCHIFY = 1
        B = rays_d.shape[0]  # batch_size
        flat_vec_shape = [B, -1, 3]
    else:
        DIM_BATCHIFY = 0
        flat_vec_shape = [-1, 3]

    rays_o = torch.reshape(rays_o, flat_vec_shape).float()
    rays_d = torch.reshape(rays_d, flat_vec_shape).float()
    # NOTE: already normalized
    rays_d = F.normalize(rays_d, dim=-1)

    batchify_query = functools.partial(
        train_util.batchify_query, chunk=netchunk, dim_batchify=DIM_BATCHIFY
    )

    # ---------------
    # Render a ray chunk
    # ---------------
    def render_rayschunk(rays_o: torch.Tensor, rays_d: torch.Tensor):
        # rays_o: [(B), N_rays, 3]
        # rays_d: [(B), N_rays, 3]

        # [(B), N_rays] x 2
        near, far = rend_util.near_far_from_sphere(
            rays_o, rays_d, r=obj_bounding_radius
        )
        if bounded_near_far:
            near, far = compute_bounded_near_far(model, rays_o, rays_d, near, far)
        if near_bypass is not None:
            near = near_bypass * torch.ones_like(near).to(device)
        if far_bypass is not None:
            far = far_bypass * torch.ones_like(far).to(device)

        if use_view_dirs:
            view_dirs = rays_d
        else:
            view_dirs = None

        prefix_batch = [B] if batched else []
        N_rays = rays_o.shape[-2]

        # ---------------
        # Sample points on the rays
        # ---------------

        # ---------------
        # Coarse Points

        # [(B), N_rays, N_samples]
        _t = torch.linspace(0, 1, N_samples).float().to(device)
        d_coarse = near * (1 - _t) + far * _t

        # ---------------
        # Up Sampling
        if samples_output == True:
            samples = {"xyz": [], "density": [], "colors": []}
        with torch.no_grad():
            _d = d_coarse
            _xyz = rays_o.unsqueeze(-2) + _d.unsqueeze(-1) * rays_d.unsqueeze(-2)
            _sdf = batchify_query(
                model.forward_density_only,
                _xyz,
            )
            _sdf = _sdf.squeeze(-1)
            for i in range(N_upsample_iters):
                prev_sdf, next_sdf = (
                    _sdf[..., :-1],
                    _sdf[..., 1:],
                )  # (...,N_samples-1)
                prev_z_vals, next_z_vals = _d[..., :-1], _d[..., 1:]
                mid_sdf = (prev_sdf + next_sdf) * 0.5
                dot_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)
                prev_dot_val = torch.cat(
                    [
                        torch.zeros_like(dot_val[..., :1], device=device),
                        dot_val[..., :-1],
                    ],
                    dim=-1,
                )  # jianfei: prev_slope, right shifted,  (...,N_samples-1)
                dot_val = torch.stack(
                    [prev_dot_val, dot_val], dim=-1
                )  # jianfei: concat prev_slope with slope ,  (...,N_samples-1,2)
                dot_val, _ = torch.min(
                    dot_val, dim=-1, keepdim=False
                )  # jianfei: find the minimum of prev_slope and current slope. (forward diff vs. backward diff., or the prev segment's slope vs. this segment's slope),  (...,N_samples-1)
                dot_val = dot_val.clamp(-10.0, 0.0)

                dist = next_z_vals - prev_z_vals
                prev_esti_sdf = mid_sdf - dot_val * dist * 0.5
                next_esti_sdf = mid_sdf + dot_val * dist * 0.5

                # phi_s_base = 64
                phi_s_base = 256
                prev_cdf = cdf_Phi_s(
                    prev_esti_sdf, phi_s_base * (2**i)
                )  # \VarPhi_s(x) in paper
                next_cdf = cdf_Phi_s(next_esti_sdf, phi_s_base * (2**i))
                alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
                _w = alpha_to_w(alpha)
                d_fine = rend_util.sample_pdf(
                    _d, _w, N_importance // N_upsample_iters, det=not perturb
                )
                _d = torch.cat([_d, d_fine], dim=-1)

                pts_fine = rays_o.unsqueeze(-2) + d_fine.unsqueeze(
                    -1
                ) * rays_d.unsqueeze(-2)
                sdf_fine = batchify_query(
                    model.forward_density_only,
                    pts_fine,
                )
                sdf_fine = sdf_fine.squeeze(-1)
                _sdf = torch.cat([_sdf, sdf_fine], dim=-1)
                _d, d_sort_indices = torch.sort(_d, dim=-1)
                _sdf = torch.gather(_sdf, DIM_BATCHIFY + 1, d_sort_indices)
            d_all = _d

        # ------------------
        # Calculate Points
        # [(B), N_rays, N_samples+N_importance, 3]
        pts = rays_o[..., None, :] + rays_d[..., None, :] * d_all[..., :, None]
        # [(B), N_rays, N_pts-1, 3]
        d_mid = 0.5 * (d_all[..., 1:] + d_all[..., :-1])
        pts_mid = rays_o[..., None, :] + rays_d[..., None, :] * d_mid[..., :, None]
        # ------------------
        # Inside Scene
        # ------------------
        if calc_normal:
            sdf, nablas = batchify_query(model.forward_with_nablas, pts)
        else:
            sdf = batchify_query(model.forward_density_only, pts)

        sdf = sdf.squeeze(-1)
        # [(B), N_ryas, N_pts], [(B), N_ryas, N_pts-1]
        cdf, opacity_alpha = sdf_to_alpha(sdf, model.forward_s())
        if random_color_direction == False:
            sdf_mid, radiances = batchify_query(
                model.forward, pts_mid, view_dirs.unsqueeze(-2).expand_as(pts_mid)
            )
        else:
            random_direction = torch.rand_like(pts_mid)
            random_direction /= torch.linalg.norm(
                random_direction, axis=-1, keepdims=True
            )
            sdf_mid, radiances = batchify_query(
                model.forward, pts_mid, random_direction
            )
        if samples_output == True:
            samples["xyz"].append(pts_mid)
            samples["density"].append(sdf_mid)
            samples["colors"].append(radiances)

        # --------------
        # Ray Integration
        # --------------
        d_final = d_mid

        # [(B), N_ryas, N_pts-1 + N_outside]
        visibility_weights = alpha_to_w(opacity_alpha)
        # [(B), N_rays]
        rgb_map = torch.sum(visibility_weights[..., None] * radiances, -2)
        # depth_map = torch.sum(visibility_weights * d_mid, -1)
        # NOTE: to get the correct depth map, the sum of weights must be 1!
        depth_map = torch.sum(
            visibility_weights
            / (visibility_weights.sum(-1, keepdim=True) + 1e-10)
            * d_final,
            -1,
        )
        acc_map = torch.sum(visibility_weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])

        ret_i = OrderedDict(
            [
                ("rgb", rgb_map),  # [(B), N_rays, 3]
                ("depth_volume", depth_map),  # [(B), N_rays]
                # ('depth_surface', d_pred_out),    # [(B), N_rays]
                ("mask_volume", acc_map),  # [(B), N_rays]
            ]
        )

        if calc_normal:
            normals_map = F.normalize(nablas, dim=-1)
            N_pts = min(visibility_weights.shape[-1], normals_map.shape[-2])
            normals_map = (
                normals_map[..., :N_pts, :] * visibility_weights[..., :N_pts, None]
            ).sum(dim=-2)
            ret_i["normals_volume"] = normals_map

        if detailed_output:
            if calc_normal:
                ret_i["implicit_nablas"] = nablas
            ret_i["implicit_surface"] = sdf
            ret_i["radiance"] = radiances
            ret_i["alpha"] = opacity_alpha
            ret_i["cdf"] = cdf
            ret_i["visibility_weights"] = visibility_weights
            ret_i["d_final"] = d_final
            if samples_output == True:
                ret_i["xyz"] = torch.cat(samples["xyz"], 2)
                ret_i["dirs"] = view_dirs.unsqueeze(-2).expand_as(ret_i["xyz"])
                ret_i["density"] = torch.cat(samples["density"], 2)
                ret_i["colors"] = torch.cat(samples["colors"], 2)

        return ret_i

    ret = {}
    for i in tqdm(
        range(0, rays_o.shape[DIM_BATCHIFY], rayschunk), disable=not show_progress
    ):
        ret_i = render_rayschunk(
            rays_o[:, i : i + rayschunk] if batched else rays_o[i : i + rayschunk],
            rays_d[:, i : i + rayschunk] if batched else rays_d[i : i + rayschunk],
        )
        for k, v in ret_i.items():
            if k not in ret:
                ret[k] = []
            ret[k].append(v)

    for k, v in ret.items():
        ret[k] = torch.cat(v, DIM_BATCHIFY)

    return ret["rgb"], ret["depth_volume"], ret


class SingleRenderer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, rays_o, rays_d, **kwargs):
        return volume_render(rays_o, rays_d, self.model, **kwargs)
