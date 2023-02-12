from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.renderer import SingleRenderer

from utils import rend_util
from utils.metric_util import *


class DensityLoss(nn.Module):
    def __init__(self, density_clip=0.1):
        super().__init__()
        self.density_clip = density_clip

    def forward(self, density_pred, density_gt):
        loss = F.l1_loss(density_gt, density_pred, reduction="none")
        mask = density_gt.abs() <= self.density_clip
        # we assume at least one density is smaller than density_clip
        loss = loss[mask].mean()
        return loss


class Trainer(nn.Module):
    def __init__(
        self,
        model,
        loss_weights,
        teacher_model=None,
        device_ids=[0],
        batched=True,
    ):
        super().__init__()
        self.model = model
        self.device = device_ids[0]
        self.renderer = SingleRenderer(self.model)
        if len(device_ids) > 1:
            self.renderer = nn.DataParallel(
                self.renderer, device_ids=device_ids, dim=1 if batched else 0
            )
        self.teacher_model = teacher_model
        if self.teacher_model != None:
            self.teacher_model.to(self.device).eval()

        self.loss_weights = loss_weights
        self.density_loss = DensityLoss()

    def forward(
        self,
        args,
        indices,
        model_input,
        ground_truth,
        render_kwargs_train: dict,
        it: int,
        train_progress: float = 0,  # from 0 to 1
        device="cuda",
    ):

        intrinsics = model_input["intrinsics"].to(device)
        c2w = model_input["c2w"].to(device)
        H = render_kwargs_train["H"]
        W = render_kwargs_train["W"]
        rays_o, rays_d, select_inds = rend_util.get_rays(
            c2w, intrinsics, H, W, N_rays=args.data.N_rays
        )

        use_distill_loss = (
            self.loss_weights["distill_density"] > 0
            or self.loss_weights["distill_color"] > 0
        )

        rgb, depth_v, extras = self.renderer(
            rays_o,
            rays_d,
            detailed_output=True,
            samples_output=use_distill_loss,
            **render_kwargs_train,
        )

        use_eikonal_loss = (
            "implicit_nablas" in extras and self.loss_weights["eikonal"] > 0
        )
        use_mask = self.loss_weights["mask"] > 0
        use_indicator_reg = self.loss_weights["indicator_reg"] > 0
        # [B, N_rays, 3]
        target_rgb = torch.gather(
            ground_truth["rgb"].to(device), 1, torch.stack(3 * [select_inds], -1)
        )
        target_mask = (
            torch.gather(model_input["object_mask"].to(device), 1, select_inds)
            if use_mask
            else None
        )
        mask_ignore = (
            torch.gather(model_input["mask_ignore"].to(device), 1, select_inds)
            if "mask_ignore" in model_input
            else None
        )
        ret = self.compute_loss(
            args,
            rgb,
            target_rgb,
            extras,
            mask=target_mask,
            mask_ignore=mask_ignore,
            use_distill_loss=use_distill_loss,
            use_eikonal_loss=use_eikonal_loss,
            use_indicator_reg=use_indicator_reg,
        )
        ret["extras"]["select_inds"] = select_inds

        return ret

    def forward_painting(
        self,
        args,
        indices,
        model_input,
        ground_truth,
        render_kwargs_train: dict,
        it: int,
        train_progress: float = 0,  # from 0 to 1
        device="cuda",
    ):
        def render_rays(affix_name, samples_output=False, random_direction=False):
            rays_o = model_input["rays_o_" + affix_name]
            rays_d = model_input["rays_d_" + affix_name]
            is_painting = torch.ones_like(model_input["mask_" + affix_name])
            mask = model_input["mask_" + affix_name]
            target_rgb = ground_truth["rgb_" + affix_name]
            rays_o = rays_o.unsqueeze(1).to(device)
            rays_d = rays_d.unsqueeze(1).to(device)
            is_painting = is_painting.unsqueeze(1).to(device)
            mask = mask.unsqueeze(1).to(device)
            target_rgb = target_rgb.unsqueeze(1).to(device)
            rgb, depth_v, extras = self.renderer(
                rays_o,
                rays_d,
                detailed_output=True,
                samples_output=samples_output,
                random_color_direction=random_direction,
                **render_kwargs_train,
            )
            return rgb, target_rgb, mask, extras

        (
            rgb_paint,
            target_rgb_paint,
            mask_paint,
            extras_paint,
        ) = render_rays("paint", random_direction=True)
        rgb_bg, target_rgb_bg, mask_bg, extras_bg = render_rays(
            "bg", samples_output=True
        )
        rgb = torch.cat([rgb_paint, rgb_bg], dim=0)
        target_rgb = torch.cat([target_rgb_paint, target_rgb_bg], dim=0)
        mask = torch.cat([mask_paint, mask_bg], dim=0)
        extras_bg["mask_volume"] = torch.cat(
            [extras_bg["mask_volume"], extras_paint["mask_volume"]], dim=0
        )
        return self.compute_loss(
            args,
            rgb,
            target_rgb,
            extras_bg,
            mask=mask,
            use_distill_loss=True,
        )

    def compute_loss(
        self,
        args,
        rgb,
        target_rgb,
        extras,
        mask=None,
        mask_ignore=None,
        use_eikonal_loss=False,
        use_distill_loss=False,
        use_indicator_reg=False,
    ):

        if use_eikonal_loss:
            # [B, N_rays, N_pts, 3]
            nablas: torch.Tensor = extras["implicit_nablas"]
            # [B, N_rays, N_pts]
            nablas_norm = torch.norm(nablas, dim=-1)
        # [B, N_rays]
        mask_volume: torch.Tensor = extras["mask_volume"]
        # NOTE: when predicted mask is close to 1 but GT is 0, exploding gradient.
        mask_volume = torch.clamp(mask_volume, 1e-3, 1 - 1e-3)
        extras["mask_volume_clipped"] = mask_volume

        losses = OrderedDict()

        # [B, N_rays, 3]
        losses["loss_img"] = self.loss_weights["img"] * F.l1_loss(
            rgb, target_rgb, reduction="none"
        )
        # [B, N_rays, N_pts]
        if use_eikonal_loss:
            losses["loss_eikonal"] = self.loss_weights["eikonal"] * F.mse_loss(
                nablas_norm,
                nablas_norm.new_ones(nablas_norm.shape),
                reduction="mean",
            )

        if use_distill_loss:
            with torch.no_grad():
                gt_sdf, gt_radiances = self.teacher_model(extras["xyz"], extras["dirs"])
            losses["loss_density"] = self.loss_weights["distill_density"] * F.l1_loss(
                extras["density"], gt_sdf.unsqueeze(-1), reduction="mean"
            )
            losses["loss_color"] = self.loss_weights["distill_color"] * F.mse_loss(
                extras["colors"], gt_radiances, reduction="mean"
            )
        if use_indicator_reg:
            losses["loss_indicator_vector_reg"] = self.loss_weights["indicator_reg"] * (
                F.mse_loss(
                    self.model.indicator_vector,
                    self.model.mesh_grid.get_vertex_normal_torch(),
                ).mean()
            )
        if mask is not None:
            # [B, N_rays]
            target_mask = mask
            losses["loss_mask"] = self.loss_weights["mask"] * F.binary_cross_entropy(
                mask_volume, target_mask.float(), reduction="mean"
            )
            if mask_ignore is not None:
                target_mask = torch.logical_and(target_mask, mask_ignore)

            # [N_masked, 3]
            losses["loss_img"] = (
                losses["loss_img"] * target_mask[..., None].float()
            ).sum() / (target_mask.sum() + 1e-10)

            extras["psnr"] = psnr(
                rgb[target_mask],
                target_rgb[target_mask],
            )
        else:
            if mask_ignore is not None:
                losses["loss_img"] = (
                    losses["loss_img"] * mask_ignore[..., None].float()
                ).sum() / (mask_ignore.sum() + 1e-10)
                extras["psnr"] = psnr(
                    rgb[mask_ignore],
                    target_rgb[mask_ignore],
                    reduction="none",
                )
            else:
                losses["loss_img"] = losses["loss_img"].mean()
                extras["psnr"] = psnr(rgb, target_rgb)

        loss = 0
        for k, v in losses.items():
            loss += losses[k]

        losses["total"] = loss
        if use_eikonal_loss:
            extras["implicit_nablas_norm"] = nablas_norm
        extras["scalars"] = {"1/s": 1.0 / self.model.forward_s().data}
        if use_indicator_reg and self.model.learn_indicator_weight:
            extras["scalars"][
                "indicator_weight"
            ] = self.model.forward_indicator_weight().data

        return OrderedDict([("losses", losses), ("extras", extras)])
