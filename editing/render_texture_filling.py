import argparse
from scipy import spatial
import open3d as o3d
import numpy as np

import torch

from utils import io_util
from utils.vis_mesh_util import preview_transfer_on_mesh

from editing.texture_neumesh.editable_primitive import EditingParams
from editing.texture_neumesh.texture_renderer import TextureEditableRenderer

from render import create_render_args


def find_nearest_neighbour(mesh1, mesh2, EPS=1e-6):
    """
    find the closest vertex of mesh1 to each vertex of mesh2
    """
    vertices1 = np.asarray(mesh1.vertices).astype(np.float32)
    vertices2 = np.asarray(mesh2.vertices).astype(np.float32)
    tree2 = spatial.cKDTree(vertices2)
    distance, neighbors_in_mesh2 = tree2.query(vertices1, k=1)
    assert np.all(
        distance < EPS
    ), "[Error] Misalignment exits between two meshes. Please ensure two meshes are identical, or increase the `EPS` ( #distance max: {}, min{} )".format(
        distance.max(), distance.mean()
    )
    return neighbors_in_mesh2


def invert_neighbours(neighbors_in_1, mask0):
    neighbors_in_0 = {}  # neighbors_in_0[i1] = [i0,...]
    for i0 in range(len(neighbors_in_1)):
        i1 = neighbors_in_1[i0]
        if mask0[i0] == False:
            continue
        if i1 not in neighbors_in_0:
            neighbors_in_0[i1] = [i0]
        else:
            neighbors_in_0[i1].append(i0)
    return neighbors_in_0


def collect_modeluv(neighbours_in_modelmesh, triangles, triangles_uv):
    """
    get the sets (uv, model_vi)
    """
    uv_set = set()
    model_uv = []
    model_indices_of_uv = []
    for i in range(triangles.shape[0]):
        for j in range(triangles.shape[1]):
            uv = triangles_uv[i][j]
            mask_vi = triangles[i][j]
            if mask_vi in neighbours_in_modelmesh:
                for model_vi in neighbours_in_modelmesh[mask_vi]:
                    uv_set.add((uv[0], uv[1], model_vi))
    for uvi in uv_set:
        model_uv.append([uvi[0], uvi[1]])
        model_indices_of_uv.append(uvi[2])
    print("[Info] after clean: {}".format(len(model_uv)))
    return np.array(model_uv), np.array(model_indices_of_uv)


class TextureFillingRender(TextureEditableRenderer):
    def __init__(self):
        super().__init__()

    def read_editing_mask(self, mask_path, mesh):
        mask_mesh = o3d.io.read_triangle_mesh(mask_path)
        N_model = len(mesh.vertices)
        N_mask = len(mask_mesh.vertices)
        neighbors_in_maskmesh = find_nearest_neighbour(
            mesh, mask_mesh
        )  # (N_model), neighbors_in_maskmesh[model_vi] = mask_vi
        triangles_uv = (
            np.array(mask_mesh.triangle_uvs).astype(np.float32).reshape(-1, 3, 2)
        )  # (N_mask, 3,2)
        maskmesh_triangles = np.asarray(mask_mesh.triangles)  # (N_mask, 3)
        editing_triangles_mask = (
            np.linalg.norm(triangles_uv, axis=-1) > 1e-8
        )  # (N_mask, 3)
        editing_vertices_mask = np.int32(np.zeros(N_mask))  # (N_mask)
        editing_vertices_mask[maskmesh_triangles[editing_triangles_mask]] = 1
        mask = editing_vertices_mask[neighbors_in_maskmesh] == 1  # (N_model)

        neighbors_in_modelmesh = invert_neighbours(neighbors_in_maskmesh, mask)
        model_uv, model_indices_of_uv = collect_modeluv(
            neighbors_in_modelmesh, maskmesh_triangles, triangles_uv
        )  # (N_model, 2), (N_model)
        editing_params = EditingParams(mask, model_uv, model_indices_of_uv)

        return editing_params

    def transfer_texture_features(
        self,
        args,
        main_primitive,
        ref_primitives,
    ):
        for i in range(len(ref_primitives)):
            main_editing_params = main_primitive.get_editing_params(i)
            ref_primitive = ref_primitives[i]
            ref_editing_params = ref_primitive.get_editing_params(0)

            main_editing_params.clamp_and_normalize_params()
            ref_editing_params.clamp_and_normalize_params()

            self.transfer(
                main_primitive,
                main_editing_params,
                ref_primitive,
                ref_editing_params,
                steps=args.step[i],
                Kc=args.Kc,
                debug_draw=args.debug_draw,
            )

        if args.debug_draw == True:
            exit(0)

    def transfer(
        self,
        main_primitive,
        main_params,
        ref_primitive,
        ref_params,
        steps=1,
        Kc=4,
        debug_draw=False,
    ):

        weights, ref_feat_indices, main_feat_indices = self.compute_transition_weights(
            main_params, ref_params, steps, Kc
        )

        ref_feat = ref_primitive.model.color_features[
            ref_feat_indices
        ]  # (Nm, Kc, fc_dim)
        new_main_feat = torch.sum(
            weights.unsqueeze(-1) * ref_feat, dim=-2
        )  # (Nm, fc_dim)
        main_primitive.edit_color_features[main_feat_indices] = new_main_feat

        if debug_draw == True:
            preview_transfer_on_mesh(
                main_primitive.get_mesh(),
                ref_primitive.get_mesh(),
                ref_feat_indices,
                weights.numpy(),
                main_feat_indices,
            )

    def compute_transition_weights(self, main_params, ref_params, steps, Kc):
        # convolve the reference uv onto main uv
        mainuv_size = main_params.get_size_of_uv()
        refuv_size = ref_params.get_size_of_uv()
        dimension = np.argmax(refuv_size)  # select the longer side
        ref_scale = mainuv_size[dimension] / (steps * refuv_size[dimension])
        kernel_size = refuv_size * ref_scale
        coord = main_params.get_uv() / kernel_size  # (Nm, 2)
        coord_in_kernel = ((coord - np.int32(coord)) * kernel_size) / ref_scale

        # search the Kc closest neighbour in reference uv
        refuv_tree = spatial.cKDTree(ref_params.get_uv().reshape(-1, 2))
        distance, neighbours_in_refuv = refuv_tree.query(
            coord_in_kernel, k=Kc
        )  # (Nm, Kc)

        # compute new feature for each uv
        weights_t = 1 / (distance + 1e-8)  # inverse depth
        weights_t = torch.FloatTensor(
            weights_t / np.sum(weights_t, axis=-1, keepdims=True)
        )  # (Nm, Kc)

        return (
            weights_t,
            ref_params.get_vertex_ind_of_uv()[neighbours_in_refuv],
            main_params.get_vertex_ind_of_uv(),
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--Kc", default=4, type=int)
    parser.add_argument("--debug_draw", action="store_true")
    create_render_args(parser)
    args, unknown = parser.parse_known_args()
    config_dict = io_util.read_json(args.config)
    other_dict = vars(args)
    config_dict.update(other_dict)
    config = io_util.ForceKeyErrorDict(**config_dict)
    renderer = TextureFillingRender()
    renderer.forward(config)
