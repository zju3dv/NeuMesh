import numpy as np
import torch
import torch.nn as nn

from models.base import ImplicitSurface, RadianceNet, NeRF


class NeuS(nn.Module):
    def __init__(
        self,
        variance_init=0.05,
        speed_factor=1.0,
        input_ch=3,
        W_geo_feat=-1,
        use_outside_nerf=False,
        obj_bounding_radius=1.0,
        surface_cfg=dict(),
        radiance_cfg=dict(),
    ):
        super().__init__()

        self.ln_s = nn.Parameter(
            data=torch.Tensor([-np.log(variance_init) / speed_factor]),
            requires_grad=True,
        )
        self.speed_factor = speed_factor

        # ------- surface network
        self.implicit_surface = ImplicitSurface(
            W_geo_feat=W_geo_feat,
            input_ch=input_ch,
            obj_bounding_size=obj_bounding_radius,
            **surface_cfg
        )

        # ------- radiance network
        if W_geo_feat < 0:
            W_geo_feat = self.implicit_surface.W
        self.radiance_net = RadianceNet(W_geo_feat=W_geo_feat, **radiance_cfg)

        # -------- outside nerf++
        if use_outside_nerf:
            self.nerf_outside = NeRF(
                input_ch=4, multires=10, multires_view=4, use_view_dirs=True
            )

    def forward_radiance(
        self, x: torch.Tensor, view_dirs: torch.Tensor
    ):  # Q(BCH): what is the output of radiance_net? A : color
        _, nablas, geometry_feature = self.implicit_surface.forward_with_nablas(
            x
        )  # Comment(BCHO): nablas is the normal of surface resulted from gradient of sdf
        radiance = self.radiance_net.forward(
            x, view_dirs, nablas, geometry_feature
        )  # Q(BCH): geometry_feature is the embedding of x, and why is it needed feed again?
        return radiance

    def forward_s(self):
        return torch.exp(self.ln_s * self.speed_factor)

    def forward(self, x: torch.Tensor, view_dirs: torch.Tensor):
        sdf, nablas, geometry_feature = self.implicit_surface.forward_with_nablas(x)
        radiances = self.radiance_net.forward(x, view_dirs, nablas, geometry_feature)
        return sdf, radiances

    def forward_density_only(self, x: torch.Tensor):
        return self.implicit_surface.forward(x)

    def forward_with_nablas(self, x: torch.Tensor, has_grad_bypass: bool = None):
        return self.implicit_surface.forward_with_nablas(x, has_grad_bypass)[:2]
