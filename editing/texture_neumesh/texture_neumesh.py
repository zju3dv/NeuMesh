import torch
import torch.nn as nn

from utils.geo_util import transform_direction


class TextureEditableNeuMesh(nn.Module):
    def __init__(
        self,
        main_model,
        ref_models,
        main_editing_masks,
        main_editing_colorfeats,
        T_r_m_list=None,
    ):
        super(TextureEditableNeuMesh, self).__init__()
        self.main_model = main_model
        self.ref_models = nn.ModuleList(ref_models)
        self.register_buffer("main_editing_masks", main_editing_masks)
        self.register_buffer("main_editing_colorfeats", main_editing_colorfeats)
        if T_r_m_list is not None:
            rot_s_m_list = []
            t_s_m_list = []
            for T_r_m in T_r_m_list:
                rot_s_m_list.append(T_r_m[:3, :3])
                t_s_m_list.append(T_r_m[:3, 3])
            self.register_buffer(
                "rot_s_m",
                torch.stack(rot_s_m_list, dim=0),
            )
            self.register_buffer(
                "t_s_m",
                torch.stack(t_s_m_list, dim=0),
            )
        else:
            self.rot_s_m = None
            self.t_s_m = None

        self.enable_nablas_input = main_model.enable_nablas_input

    def compute_distance(self, xyz):
        return self.main_model.compute_distance(xyz)

    def forward_s(self):
        return self.main_model.forward_s()

    def forward_density_only(self, xyz):
        return self.main_model.forward_density_only(xyz)

    def forward_with_nablas(self, xyz: torch.Tensor):
        return self.main_model.forward_with_nablas(xyz)

    def forward(
        self,
        xyz: torch.Tensor,
        view_dirs: torch.Tensor,
        need_nablas=True,
        nablas_only=False,
    ):
        """
        xyz: (...,3)
        dirs: (...,3)
        """
        sdf, nabla, _ds, _indices, _weights = self.main_model.forward(
            xyz,
            view_dirs,
            need_nablas=need_nablas,
            nablas_only=True,
            return_ds=True,
        )
        colors = self.main_model.forward_color(
            _ds,
            view_dirs,
            self.main_model.color_features,
            indices=_indices,
            weights=_weights,
            nabla=nabla,
        )

        blend_color = colors.clone()
        for i in range(len(self.ref_models)):
            ref_model = self.ref_models[i]
            main_mask = self.main_editing_masks[i]
            # b. compute blending weights
            paint_weight = torch.sum(_weights * main_mask[_indices], dim=-1)
            unpaint_weight = torch.sum(
                _weights * (main_mask[_indices] == False), dim=-1
            )
            paint_region = paint_weight > 0
            sum_weight = paint_weight + unpaint_weight
            paint_weight /= sum_weight
            unpaint_weight /= sum_weight
            paint_weight = paint_weight[paint_region]
            unpaint_weight = unpaint_weight[paint_region]

            ref_weights = _weights * main_mask[_indices]
            ref_weights /= torch.sum(ref_weights, dim=-1, keepdim=True) + 1e-8

            # c. query slave color
            if self.rot_s_m is not None:
                rot_s_m = self.rot_s_m[i]
                ref_dir = transform_direction(rot_s_m, view_dirs)
                ref_nabla = transform_direction(rot_s_m, nabla)
            else:
                ref_dir = view_dirs
                ref_nabla = nabla
            if torch.any(paint_region) == True:
                ref_color = ref_model.forward_color(
                    _ds[paint_region],
                    ref_dir[paint_region],
                    self.main_editing_colorfeats,
                    indices=_indices[paint_region],
                    weights=ref_weights[paint_region],
                    nabla=ref_nabla[paint_region],
                )
                blend_color[paint_region] = blend_color[
                    paint_region
                ] * unpaint_weight.unsqueeze(-1) + ref_color * paint_weight.unsqueeze(
                    -1
                )

        return sdf, blend_color
