import argparse
import open3d as o3d
import numpy as np
import json
from scipy import spatial
import copy

import torch

from .texture_neumesh.editable_primitive import EditingParams
from .texture_neumesh.texture_renderer import TextureEditableRenderer

from render import create_render_args
from tools import interactive_mesh_algnment

from utils import io_util
from utils.mesh_util import (
    check_degenerate_triangles,
    get_isolated_mask,
    clean_duplicate_triangles,
)
from utils.geo_util import transform_vertices
from utils.vis_mesh_util import preview_transfer_on_mesh


def interactive_rigid_transform_mesh(main_mesh, ref_mesh):
    main_pcd = o3d.geometry.PointCloud()
    main_pcd.points = main_mesh.vertices
    main_pcd.colors = main_mesh.vertex_colors
    ref_pcd = o3d.geometry.PointCloud()
    ref_pcd.points = ref_mesh.vertices
    ref_pcd.colors = ref_mesh.vertex_colors
    corr, T_r_m = interactive_mesh_algnment.demo_manual_registration(main_pcd, ref_pcd)
    return corr, T_r_m


def deform_mesh_func(pt1_trans, corr, ref_mesh, ref_mask):
    check_degenerate_triangles(ref_mesh)
    vertices = np.asarray(ref_mesh.vertices)
    isolated_mask = get_isolated_mask(ref_mesh)
    static_ids = np.where(np.logical_or(ref_mask == False, isolated_mask == True))[0]
    static_pos = np.array([vertices[i] for i in static_ids])
    handle_ids = corr
    handle_pos = pt1_trans

    if static_ids.shape[0] == 0:
        constraint_ids = o3d.utility.IntVector(handle_ids.astype(np.int32))
        constraint_pos = o3d.utility.Vector3dVector(handle_pos)
    else:
        constraint_ids = o3d.utility.IntVector(
            np.concatenate([static_ids, handle_ids], axis=0).astype(np.int32)
        )
        constraint_pos = o3d.utility.Vector3dVector(
            np.concatenate([static_pos, handle_pos], axis=0)
        )
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh_prime = ref_mesh.deform_as_rigid_as_possible(
            constraint_ids, constraint_pos, max_iter=20
        )
    ref_mesh.vertices = mesh_prime.vertices


def save_rigid_transform(args, T_r_m_list, corr_list):
    with open(args.config, "r") as f:
        data = json.loads(f.read())
        data["T_r_m"] = np.array(T_r_m_list).tolist()
        data["corr"] = [a.tolist() for a in corr_list]
    with open(args.config, "w") as f:
        json.dump(data, f, indent=2)


def align_mesh(
    main_mesh,
    main_editing_mask,
    ref_mesh,
    ref_editing_mask,
    T_r_m=None,
    corr=None,
    use_arap=False,
):
    # align meshes by affine transformation
    if T_r_m is None or corr is None:
        # black the unedited region
        main_vis_mesh = copy.deepcopy(main_mesh)
        main_color = np.array(main_vis_mesh.vertex_colors)
        main_color[main_editing_mask == False] = 0
        main_vis_mesh.vertex_colors = o3d.utility.Vector3dVector(main_color)
        ref_vis_mesh = copy.deepcopy(ref_mesh)
        ref_color = np.array(ref_vis_mesh.vertex_colors)
        ref_color[ref_editing_mask == False] = 0
        ref_vis_mesh.vertex_colors = o3d.utility.Vector3dVector(ref_color)
        # interactive rigid transform
        corr, T_r_m = interactive_rigid_transform_mesh(main_vis_mesh, ref_vis_mesh)
        corr = np.int32(corr)

    # align meshes by as-rigid-as-possible
    if use_arap == True:
        clean_duplicate_triangles(ref_mesh)
        pt1 = np.asarray(main_mesh.vertices)[corr[:, 0]]
        pt1_trans = (T_r_m[:3, :3] @ pt1[..., None]).squeeze(-1) + T_r_m[:3, 3]
        deform_mesh_func(pt1_trans, corr[:, 1], ref_mesh, ref_editing_mask)
    return T_r_m, corr, ref_mesh


class TextureSwappingRender(TextureEditableRenderer):
    def __init__(self):
        super().__init__()

    def read_editing_mask(self, mask_path, mesh):
        mask_mesh = o3d.io.read_triangle_mesh(mask_path)
        mask = np.sum(np.asarray(mask_mesh.vertex_colors), axis=-1) != 0
        editing_params = EditingParams(mask)

        return editing_params

    def transfer_texture_features(self, args, main_primitive, ref_primitives):

        # transfer color codes
        T_r_m_list = []
        corr_list = []
        for i in range(len(ref_primitives)):
            main_editing_params = main_primitive.get_editing_params(i)
            ref_primitive = ref_primitives[i]
            ref_editing_params = ref_primitive.get_editing_params(0)

            if i not in args.estimate_srt and (
                len(args.T_r_m) < i + 1 or len(args.corr) < i + 1
            ):
                print(
                    "[Info] Please enable estimate_srt for {}-th editing pair".format(i)
                )
                exit(0)

            T_r_m, corr, ref_mesh_deformed = align_mesh(
                main_primitive.get_mesh(),
                main_editing_params.get_editing_mask(),
                ref_primitive.get_mesh(),
                ref_editing_params.get_editing_mask(),
                T_r_m=args.T_r_m[i] if i not in args.estimate_srt else None,
                corr=args.corr[i] if i not in args.estimate_srt else None,
                use_arap=args.use_arap,
            )
            ref_primitive.update_mesh_grid(ref_mesh_deformed)

            self.transfer(
                main_primitive,
                main_editing_params,
                ref_primitive,
                ref_editing_params,
                torch.FloatTensor(T_r_m),
                debug_draw=args.debug_draw,
            )

            T_r_m_list.append(T_r_m)
            corr_list.append(corr)

        if args.debug_draw == True and args.estimate_srt:
            save_rigid_transform(args, T_r_m_list, corr_list)
        if args.debug_draw == True:
            exit(0)

        return np.stack(T_r_m_list)

    def transfer(
        self,
        main_primitive,
        main_params,
        ref_primitive,
        ref_params,
        T_r_m,
        Kc=4,
        debug_draw=False,
    ):
        weights, ref_feat_indices, main_feat_indices = self.compute_transition_weights(
            main_primitive.get_mesh_vertices(),
            main_params,
            ref_primitive.get_mesh_vertices(),
            ref_params,
            T_r_m,
            Kc,
        )

        ref_feat = ref_primitive.model.color_features[
            ref_feat_indices
        ]  # (Nm, Kc, fg_dim)
        new_main_feat = torch.sum(
            weights.unsqueeze(-1) * ref_feat, dim=-2
        )  # (Nm, fg_dim)
        main_primitive.edit_color_features[main_feat_indices] = new_main_feat

        if debug_draw == True:
            print("[Info] preview transfer results on mesh")
            preview_transfer_on_mesh(
                main_primitive.get_mesh(),
                ref_primitive.get_mesh(),
                ref_feat_indices,
                weights.numpy(),
                main_feat_indices,
            )

    def compute_transition_weights(
        self, main_vertices, main_params, ref_vertices, ref_params, T_r_m, Kc
    ):
        # transform main vertices to reference space
        main_editing_mask = main_params.get_editing_mask()
        main_editing_indices = np.where(main_editing_mask)[0]
        ref_editing_mask = ref_params.get_editing_mask()
        ref_editing_indices = np.where(ref_editing_mask)[0]

        main_editing_vertices = main_vertices[main_editing_mask]
        ref_editing_vertices = ref_vertices[ref_editing_mask]
        main_vertices_trans = transform_vertices(
            T_r_m[:3, :3], T_r_m[:3, 3], main_editing_vertices
        )  # (Nm,3)

        # compute closest K=8 vertices for each main vertex
        ref_tree = spatial.cKDTree(ref_editing_vertices.reshape(-1, 3))
        distance, neighbours_in_refediting = ref_tree.query(
            main_vertices_trans, k=Kc
        )  # (Nm, Kc)
        neighbours_in_ref = ref_editing_indices[neighbours_in_refediting]

        # assign new code to main model
        weights_t = 1 / (distance + 1e-8)  # inverse depth
        weights_t = torch.FloatTensor(
            weights_t / np.sum(weights_t, axis=-1, keepdims=True)
        )  # (Nm, Kc)

        return weights_t, neighbours_in_ref, main_editing_indices


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--use_arap", action="store_true", default=False)
    parser.add_argument("--debug_draw", action="store_true")
    parser.add_argument(
        "--estimate_srt",
        nargs="+",
        default=[],
        type=int,
        help="estimate rigid transform for `estiamte_srt[i]` editing pair",
    )
    parser.add_argument("--fix_indicator", action="store_true", default=False)
    create_render_args(parser)
    args, unknown = parser.parse_known_args()
    config_dict = io_util.read_json(args.config)
    other_dict = vars(args)
    config_dict.update(other_dict)
    config = io_util.ForceKeyErrorDict(**config_dict)
    renderer = TextureSwappingRender()
    renderer.forward(config)
