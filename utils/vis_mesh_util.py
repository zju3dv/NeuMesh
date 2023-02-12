import open3d as o3d
import numpy as np
import copy


def vis_and_painting(mesh, mask):
    mesh_vis = copy.deepcopy(mesh)
    colors = np.array(mesh_vis.vertex_colors)
    colors[mask] = np.array([0.0, 1.0, 1.0])
    mesh_vis.vertex_colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([mesh_vis])


def preview_transfer_on_mesh(
    main_mesh, ref_mesh, neighbours_in_ref, weights, main_mask
):
    main_mesh_debug = o3d.geometry.TriangleMesh()
    main_mesh_debug = copy.deepcopy(main_mesh)
    colors_main_debug = np.asarray(main_mesh_debug.vertex_colors)
    ref_colors = np.array(ref_mesh.vertex_colors)

    colors_searched = ref_colors[neighbours_in_ref, :]  # (Nm,Kc,fg_dim)
    colors_brand = np.sum(weights[..., None] * colors_searched, axis=-2)  # (Nm, fg_dim)
    colors_main_debug[main_mask] = colors_brand
    main_mesh_debug.vertex_colors = o3d.utility.Vector3dVector(colors_main_debug)
    o3d.visualization.draw_geometries([main_mesh_debug])
