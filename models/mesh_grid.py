import numpy as np
import open3d as o3d

import torch
import frnn


class MeshPrimitive:
    def __init__(self, mesh):
        """
        Mesh base class
        Args:
            mesh: [open3d.geometry.TriangleMesh] input mesh
        Attributes:
            mesh: [open3d.geometry.TriangleMesh] input mesh
            scene: [open3d.t.geometry.RaycastingScene] the raycasting scene containing input mesh
        """
        # Note: the modification of `self.mesh` will not affect the `compute_distance`
        self.mesh = mesh
        self.mesh.compute_vertex_normals()

        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        self.scene = o3d.t.geometry.RaycastingScene()
        _ = self.scene.add_triangles(mesh_t)

    def cast_ray(self, rays_o, rays_d):
        """
        Do ray cast on the mesh
        Args:
            rays_o: (N,3) ray origins
            rays_d: (N,3) directions of rays
        Returns:
            `t_hit` is the distance to the intersection and
            `primitive_ids` is the triangle index of the triangle that was hit
        """
        rays_core = np.concatenate([rays_o, rays_d], axis=1)
        rays = o3d.core.Tensor([rays_core], dtype=o3d.core.Dtype.Float32)
        ans = self.scene.cast_rays(rays)
        return ans["t_hit"].numpy(), ans["primitive_ids"].numpy()

    def get_number_of_vertices(self):
        return len(self.mesh.vertices)


class MeshGrid(MeshPrimitive):
    def __init__(self, mesh, device, distance_method="frnn"):
        """
        Grid for the interpolated signed distance to meshes.

        Args:
            mesh: [open3d.geometry.TriangleMesh] input mesh
            device: `cpu` or `cuda`
            distance_method: [`frnn`] metric of distance
        Attributes:
            vertices: vertices of mesh
            vertex_normals : normal of vertices
            grid: A tuple of tensors consisting of cached grid structure.
        """
        super().__init__(mesh)
        self.vertices = torch.FloatTensor(np.asarray(mesh.vertices)).to(device)
        self.vertex_normals = torch.FloatTensor(np.asarray(mesh.vertex_normals)).to(
            device
        )
        _, _, _, self.grid = frnn.frnn_grid_points(
            self.vertices.unsqueeze(0),
            self.vertices.unsqueeze(0),
            None,
            None,
            K=32,
            r=100.0,
            grid=None,
            return_nn=False,
            return_sorted=True,
        )
        self.distance_method = distance_method

    def compute_distance(self, xyz, indicator_vector=None, indicator_weight=0.1, K=8):
        if self.distance_method == "frnn":
            return self.compute_distance_frnn(
                xyz,
                K,
                indicator_vector=indicator_vector,
                indicator_weight=indicator_weight,
            )
        else:
            raise NotImplementedError

    def compute_distance_frnn(
        self,
        xyz,
        K=8,
        indicator_vector=None,
        indicator_weight=0.1,
    ):
        """
        Compute the interpolated signed distance of query points to mesh

        Args:
            xyz: (N,3) query points,
            K: number of nearest neighbour considered
            indicator_vector: (N,3) learnable indicator vector of each vertex
            indicator_weight: weights of indicator vector influence

        Returns:
            distance: (N,1) the interpolated signed distance of query points to mesh
            indices: (N,K) the vertex indices of nearest K neighbours
            weights: (N,K) the weights(inverse distance) of K nearest neighbours
        """
        dis, indices, _, _ = frnn.frnn_grid_points(
            xyz.unsqueeze(0),
            self.vertices.unsqueeze(0),
            None,
            None,
            K=K,
            r=100.0,
            grid=self.grid,
            return_nn=False,
            return_sorted=True,
        )  # (1,M,K)
        # detach to make the other differentiable
        dis = dis.detach()
        indices = indices.detach()
        dis = dis.sqrt()
        weights = 1 / (dis + 1e-7)
        weights = weights / torch.sum(weights, dim=-1, keepdims=True)  # (1,M,K)
        indices = indices.squeeze(0)  # (M, K)
        weights = weights.squeeze(0)  # (M, K)

        distance = torch.zeros((xyz.shape[0], 1)).to(xyz.device)
        indicator_vec = (
            self.vertex_normals if indicator_vector is None else indicator_vector
        )
        w1 = indicator_weight
        dir_vec = xyz.unsqueeze(-2) - self.vertices[indices]
        w2 = torch.norm(dir_vec, dim=-1, keepdim=True)
        middle_vec = (indicator_vec[indices] * w1 + dir_vec * w2) / (w1 + w2)
        distance = weights.unsqueeze(-1) * torch.sum(
            dir_vec * middle_vec,
            dim=-1,
            keepdim=True,
        )
        distance = torch.sum(distance, dim=-2)

        return distance, indices, weights

    def get_vertex_normal_torch(self):
        return self.vertex_normals

    def get_vertices_torch(self):
        return self.vertices
