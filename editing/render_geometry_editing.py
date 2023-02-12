import os
import argparse
import open3d as o3d

import torch
import torch.nn as nn

from kornia.geometry.conversions import angle_axis_to_rotation_matrix

from models.frameworks import build_framework
from models.mesh_grid import MeshGrid

from utils import io_util
from utils.checkpoints import sorted_ckpts
from utils.print_fn import log

from render import create_render_args, render_function


def cos_between_vectors(x, y, do_clamp=True):
    """
    Args:
        x: (..., 3)
        y: (..., 3)
    Return:
        (...), cos theta between two vectors
    """
    cos_theta = torch.sum(x * y, dim=-1) / (
        torch.linalg.norm(x, dim=-1) * torch.linalg.norm(y, dim=-1)
    )
    if do_clamp == True:
        return torch.clamp(cos_theta, -1, 1)
    else:
        return cos_theta


def deform_model(deformed_mesh, model, device, fix_indicator=False):
    distance_method = model.mesh_grid.distance_method
    deformed_mesh_grid = MeshGrid(
        deformed_mesh,
        device,
        distance_method=distance_method,
    )
    if fix_indicator == False:
        origin_mesh_vertex_normals = model.mesh_grid.get_vertex_normal_torch()
        deform_mesh_vertex_normals = deformed_mesh_grid.get_vertex_normal_torch()
        rot_axis = torch.cross(
            origin_mesh_vertex_normals, deform_mesh_vertex_normals, dim=-1
        )  # (N, 3)
        cos_theta = cos_between_vectors(
            origin_mesh_vertex_normals, deform_mesh_vertex_normals
        )
        rot_180_mask = cos_theta == -1
        rot_rad = torch.acos(cos_theta).unsqueeze(-1)
        rot_matrix = angle_axis_to_rotation_matrix(
            rot_axis * rot_rad
        )  # (N,3,3), Note that rotvec [0,0,0] will return identity matrix

        origin_indicator = model.indicator_vector  # (N, 3)
        deform_indicator = torch.matmul(
            rot_matrix, origin_indicator.unsqueeze(-1)
        ).squeeze()  # (N, 3)
        deform_indicator[rot_180_mask] *= -1

        model.indicator_vector = nn.Parameter(deform_indicator)

    model.mesh_grid = deformed_mesh_grid


def main_function(args):
    # get student model
    main_args = io_util.load_yaml(args.main_config)
    if args.background is not None:
        main_args.model.white_bkgd = args.background == 1
    (
        model,
        trainer,
        render_kwargs_train,
        render_kwargs_test,
        render_fn,
    ) = build_framework(main_args, main_args.model.framework)

    if args.load_pt is None:
        # automatically load 'final_xxx.pt' or 'latest.pt'
        ckpt_file = sorted_ckpts(os.path.join(args.training.exp_dir, "ckpts"))[-1]
    else:
        ckpt_file = args.load_pt

    log.info("=> Use ckpt:" + str(ckpt_file))
    state_dict = torch.load(ckpt_file, map_location=args.device)
    model.load_state_dict(state_dict["model"])
    deformed_mesh = o3d.io.read_triangle_mesh(args.deformed_mesh)
    deform_model(deformed_mesh, model, args.device, fix_indicator=args.fix_indicator)

    model.to(args.device)
    args.update(dict(main_args))
    render_function(args, render_kwargs_test, render_fn, ckpt_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fix_indicator", action="store_true", default=False)
    create_render_args(parser)
    args, unknown = parser.parse_known_args()
    other_dict = vars(args)
    config_dict = io_util.read_json(args.config)
    config_dict.update(other_dict)
    config = io_util.ForceKeyErrorDict(**config_dict)
    main_function(config)
