import numpy as np
import open3d as o3d

import torch

from models.mesh_grid import MeshGrid


def get_bbox(x):
    """
    x: (...,2)
    """
    min_value = [
        x[..., 0].min(),
        x[..., 1].min(),
    ]
    max_value = [
        x[..., 0].max(),
        x[..., 1].max(),
    ]
    return np.array(min_value), np.array(max_value)


def normalize_uv(uv, keep_wh=False):
    valid_uv = uv
    domain_min = [
        valid_uv[..., 0].min(),
        valid_uv[..., 1].min(),
    ]
    domain_max = [
        valid_uv[..., 0].max(),
        valid_uv[..., 1].max(),
    ]
    if keep_wh == False:
        valid_uv[..., 0] = (valid_uv[..., 0] - domain_min[0]) / (
            domain_max[0] - domain_min[0]
        )

        valid_uv[..., 1] = (valid_uv[..., 1] - domain_min[1]) / (
            domain_max[1] - domain_min[1]
        )
    else:
        step = max(domain_max[0] - domain_min[0], domain_max[1] - domain_min[1])
        valid_uv[..., 0] = (valid_uv[..., 0] - domain_min[0]) / step
        valid_uv[..., 1] = (valid_uv[..., 1] - domain_min[1]) / step
    uv = valid_uv
    return uv


class EditingParams:
    def __init__(self, editing_mask, uv=None, vertex_ind_of_uv=None):
        self.editing_mask = editing_mask
        self.uv = uv
        self.vertex_ind_of_uv = vertex_ind_of_uv

    def clamp_params_in_uvdomain(self, min_value, max_value):
        uv = self.uv
        is_inside = (
            (uv[..., 0] >= min_value[0])
            & (uv[..., 0] <= max_value[0])
            & (uv[..., 1] >= min_value[1])
            & (uv[..., 1] <= max_value[1])
        )
        self.uv = uv[is_inside]
        self.vertex_ind_of_uv = self.vertex_ind_of_uv[is_inside]
        self.editing_mask = self.editing_mask & False
        self.editing_mask[self.vertex_ind_of_uv] = True

    def get_size_of_uv(self):
        min_value, max_value = get_bbox(self.get_uv())
        return max_value - min_value

    def get_uv(self, i=0):
        return self.uv

    def get_vertex_ind_of_uv(self):
        return self.vertex_ind_of_uv

    def normalize_uv(self, keep_wh=True):
        normalize_uv(self.uv, keep_wh)

    def clamp_and_normalize_params(
        self, min_value=[0.0, 0.0], max_value=[1.0, 1.0], keep_wh=True
    ):
        self.clamp_params_in_uvdomain(min_value, max_value)
        self.normalize_uv(keep_wh)

    def push_editing_mask_to(self, device):
        return torch.from_numpy(self.editing_mask).to(device)

    def get_editing_mask(self, b_torch=False):
        if b_torch == True:
            return torch.from_numpy(self.editing_mask)
        else:
            return self.editing_mask


class EditablePrimitive:
    def __init__(
        self,
        model,
        editing_params_list,
        color_feature_ini,
    ):
        self.model = model
        self.edit_color_features = color_feature_ini
        self.editing_params_list = editing_params_list

    def get_len_of_mask(self):
        return len(self.editing_params_list)

    def get_editing_params(self, i=0):
        return self.editing_params_list[i]

    def get_model(self):
        return self.model

    def get_editing_masks(self, b_torch=False):
        editing_masks = []
        for params in self.editing_params_list:
            editing_masks.append(params.get_editing_mask(b_torch))
        if b_torch == True:
            return torch.stack(editing_masks, dim=0)
        else:
            return np.stack(editing_masks, axis=0)

    def get_color_features(self):
        return self.edit_color_features

    def update_mesh_grid(self, mesh):
        distance_method = self.model.mesh_grid.distance_method
        device = self.model.mesh_grid.get_vertices_torch().device
        self.model.mesh_grid = MeshGrid(
            mesh,
            device,
            distance_method=distance_method,
        )

    def get_mesh(self):
        return self.model.mesh_grid.mesh

    def get_mesh_vertices(self, b_torch=False):
        if b_torch == False:
            return np.array(self.get_mesh().vertices)
        else:
            return self.model.mesh_grid.get_vertices_torch()

    def crop_uv(self, i, min_value, max_value):
        self.editing_params_list[i].crop_uv(min_value, max_value)
