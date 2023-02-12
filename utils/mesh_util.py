import numpy as np
import open3d as o3d


def check_degenerate_triangles(mesh):
    # check degenerate triangles
    vertices = np.asarray(mesh.vertices)
    for triangle in np.asarray(mesh.triangles):
        for j in range(3):
            i1 = triangle[j]
            i2 = triangle[(j + 1) % 3]
            v1 = vertices[i1]
            v2 = vertices[i2]
            dis = np.linalg.norm(v1 - v2)
            if dis < 1e-5:
                print(vertices[i1 : i2 + 1])
            assert (
                dis > 1e-5
            ), f"vertex {triangle[j]} and {triangle[(j + 1) % 3]} is too close: {dis}, v1: {v1}, v2:{v2}"


def clean_duplicate_triangles(mesh):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    v0 = vertices[triangles[:, 0]]  # (M,3)
    v1 = vertices[triangles[:, 1]]  # (M,3)
    v2 = vertices[triangles[:, 2]]  # (M,3)

    dis1 = np.linalg.norm(v0 - v1, axis=-1)  # (M)
    dis2 = np.linalg.norm(v1 - v2, axis=-1)  # (M)
    dis3 = np.linalg.norm(v2 - v0, axis=-1)  # (M)

    valid = (dis1 > 1e-5) & (dis2 > 1e-5) & (dis3 > 1e-5)
    triangles = triangles[valid, :]
    mesh.triangles = o3d.utility.Vector3iVector(triangles)


def get_isolated_mask(mesh):
    triangles = np.asarray(mesh.triangles)

    mask = np.zeros(np.asarray(mesh.vertices).shape[0])
    used_vertices = triangles.flatten()
    mask[used_vertices] = 1
    return mask == 0
