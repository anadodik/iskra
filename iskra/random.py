# Copyright (c) 2022 - present, Ana Dodik. All rights reserved.

from dataclasses import dataclass

import torch

from iskra.geometry import tetrahedron_volumes, triangle_areas
from iskra.logging.logging import getLogger

LOGGER = getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class SamplingRecord:
    """Contains the result of a geometric sampling operation.

    Attributes:
        point (torch.Tensor): A [n_samples, D] tensor of sample positions.
        pdf (torch.Tensor): A [n_samples, 1] tensor of sample pdfs.
    """

    point: torch.Tensor
    pdf: torch.Tensor


@dataclass(kw_only=True, slots=True)
class MeshSamplingRecord:
    """Contains the result of a geometric sampling operation.

    Attributes:
        point (torch.Tensor): A [n_samples, D] tensor of sample positions.
        pdf (torch.Tensor): A [n_samples, 1] tensor of sample pdfs.
        bary (torch.Tensor): A [n_samples, n_simplex_vertices] tensor of sample
            barycentric coordinates.
        bary (torch.Tensor): A [n_samples, 1] tensor of the index of the
            sampled geometric primitive.
    """

    point: torch.Tensor
    pdf: torch.Tensor
    bary: torch.Tensor
    prim_idx: torch.Tensor


class EdgeSampler:
    def __init__(self, edges: torch.Tensor, seed: int | None = 620):
        """Uniformly randomly samples a set of line segments.

        Args:
            edges (torch.Tensor): A tensor of shape [E, 2, D] containing
                the D-dimensional positions of the line segment vertices.
            seed (int, optional): A seed for the random number generator.
                Defaults to 620.
        """
        self.edges = edges
        self.centered_edges = self.edges[:, 1, :] - self.edges[:, 0, :]
        self.edge_lengths = torch.linalg.vector_norm(
            self.centered_edges, dim=-1, keepdim=False
        )
        self.cdf = torch.cumsum(self.edge_lengths, dim=0)
        if self.cdf.shape[0] == 0:
            total_sum = 1.0
        else:
            total_sum = self.cdf[-1].clone()
        self.inv_total_length = 1 / total_sum
        self.cdf *= self.inv_total_length
        self.generator = torch.Generator(device=self.edges.device)
        if seed is not None:
            self.generator.manual_seed(seed)

    def sample(self, n_samples: int) -> MeshSamplingRecord:
        """Returns random samples uniformly distributed on the line segments.

        Args:
            n_samples (int): Number of samples to draw.

        Returns:
            SamplingRecord: Sampling record containing information about the
                sampled points.
        """
        # TODO(anadodik): Replace with torch.multinomial:
        u = torch.empty([n_samples], device=self.cdf.device).uniform_(
            0, 1, generator=self.generator
        )
        edge_idx = torch.searchsorted(self.cdf, u)
        t = torch.empty([n_samples, 1], device=self.cdf.device).uniform_(
            0, 1, generator=self.generator
        )
        bary = torch.cat([1 - t, t], -1)
        LOGGER.info(f"edge_idx={edge_idx.shape}, t={t.shape}")
        points = self.centered_edges[edge_idx, :] * t + self.edges[edge_idx, 0, :]
        pdf = self.inv_total_length.clone().to(device=points.device)
        pdf = pdf[None, None].repeat(n_samples, 1)

        samples = MeshSamplingRecord(
            point=points, pdf=pdf, bary=bary, prim_idx=edge_idx
        )

        # TODO(anadodik): remove following lines when after documenting this somewhere
        # uv = (1 - t) * self.edge_uvs[edge_idx, 0, :] + t * self.edge_uvs[edge_idx, 1, :]  # noqa: E501
        # normal = self.mesh.boundary_edge_normals[edge_idx]
        return samples


class TriangleSampler:
    def __init__(self, triangles: torch.Tensor, seed: int | None = 620):
        """Uniformly samples a set of triangles.

        Uses the procedure outlined in PBRT, Section 13.6.5, "Sampling a Triangle".

        Args:
            triangles (torch.Tensor): A tensor of shape [F, 3, D] containing
                the D-dimensional positions of the triangle vertices.
            seed (int, optional): A seed for the random number generator.
        """
        self.triangles = triangles
        self.triangle_areas = triangle_areas(triangles).flatten()
        self.cdf = torch.cumsum(self.triangle_areas, dim=0)
        self.inv_total_area = 1 / self.cdf[-1].clone()
        self.cdf *= self.inv_total_area
        self.generator = torch.Generator(device=self.triangles.device)
        if seed is not None:
            self.generator.manual_seed(seed)

    def sample(self, n_samples: int) -> MeshSamplingRecord:
        """Returns random samples uniformly distributed on the triangles.

        Args:
            n_samples (int): Number of samples to draw.

        Returns:
            SamplingRecord: Sampling record containing information about the
                sampled points.
        """
        # TODO(anadodik): Replace with torch.multinomial:
        u = torch.empty([n_samples], device=self.cdf.device).uniform_(
            0, 1, generator=self.generator
        )
        triangle_idx = torch.searchsorted(self.cdf, u)
        v = torch.empty([n_samples, 2], device=self.cdf.device).uniform_(
            0, 1, generator=self.generator
        )
        sqrt_v0 = torch.sqrt(v[:, 0])
        bary_0: torch.Tensor = 1 - sqrt_v0
        bary_1: torch.Tensor = v[:, 1] * sqrt_v0
        bary_2: torch.Tensor = 1 - (bary_0 + bary_1)
        bary = torch.stack([bary_0, bary_1, bary_2], -1)
        points = (
            bary_0[:, None] * self.triangles[triangle_idx, 0, :]
            + bary_1[:, None] * self.triangles[triangle_idx, 1, :]
            + bary_2[:, None] * self.triangles[triangle_idx, 2, :]
        )
        pdf = self.inv_total_area.clone().to(device=points.device)
        pdf = pdf[None, None].repeat(n_samples, 1)

        samples = MeshSamplingRecord(
            point=points, pdf=pdf, bary=bary, prim_idx=triangle_idx
        )
        return samples


class TetrahedronSampler:
    def __init__(self, tets: torch.Tensor, seed: int | None = 620) -> None:
        """Uniformly randomly samples a set of tetrahedra.

        Args:
            tets (torch.Tensor): A tensor of shape [T, 4, D] containing
                the D-dimensional positions of the tetrahedron vertices.
            seed (int, optional): A seed for the random number generator.
                Defaults to 620.
        """
        self.tets: torch.Tensor = tets
        volumes = torch.abs(tetrahedron_volumes(tets))
        self.inv_total_volume: torch.Tensor = 1 / torch.sum(volumes)
        self.pdfs: torch.Tensor = (volumes * self.inv_total_volume).flatten()
        self.generator = torch.Generator(device=self.tets.device)
        if seed is not None:
            self.generator.manual_seed(seed)

    def sample(self, n_samples: int) -> MeshSamplingRecord:
        """Returns random samples uniformly distributed on the triangles.

        Args:
            n_samples (int): Number of samples to draw.

        Returns:
            SamplingRecord: Sampling record containing information about the
                sampled points.
        """
        tet_idx = torch.multinomial(
            self.pdfs, num_samples=n_samples, replacement=True, generator=self.generator
        )
        u = torch.rand(n_samples, 4, device=tet_idx.device)
        bary = -torch.log(u)
        bary = bary / torch.sum(bary, dim=1, keepdim=True)
        points = (
            bary[:, 0:1] * self.tets[tet_idx, 0, :]
            + bary[:, 1:2] * self.tets[tet_idx, 1, :]
            + bary[:, 2:3] * self.tets[tet_idx, 2, :]
            + bary[:, 3:4] * self.tets[tet_idx, 3, :]
        )
        pdf = self.inv_total_volume.clone().to(device=points.device)
        pdf = pdf[None, None].repeat(n_samples, 1)

        samples = MeshSamplingRecord(point=points, pdf=pdf, bary=bary, prim_idx=tet_idx)
        return samples


def create_simplex_sampler(
    simplices: torch.Tensor, seed: int | None = 620
) -> EdgeSampler | TriangleSampler | TetrahedronSampler:
    dim = simplices.shape[-2]
    if dim == 2:
        return EdgeSampler(simplices, seed)
    elif dim == 3:
        return TriangleSampler(simplices, seed)
    elif dim == 4:
        return TetrahedronSampler(simplices, seed)
    else:
        raise NotImplementedError(
            f"Random sampling of {dim}-dimensional simplices is not supported."
        )


class BBoxSampler:
    def __init__(
        self, box_min: torch.Tensor, box_max: torch.Tensor, seed: int | None = 620
    ):
        """Uniformly samples a set of triangles.

        Uses the procedure outlined in PBRT, Section 13.6.5, "Sampling a Triangle".

        Args:
            box_min (torch.Tensor): A 1D tensor with the min corner of the bounding box.
            box_max (torch.Tensor): A 1D tensor with the max corner of the bounding box.
            seed (int, optional): A seed for the random number generator.
                Defaults to 620.
        """
        self.min = box_min
        self.max = box_max
        self.pdf = 1 / torch.prod(self.extent)[None]
        self.dim = self.extent.shape[0]
        self.generator = torch.Generator(device=self.min.device)
        if seed is not None:
            self.generator.manual_seed(seed)

    @property
    def extent(self):
        return self.max - self.min

    def sample(self, n_samples: int) -> SamplingRecord:
        """Returns random samples uniformly in a box.

        Args:
            n_samples (int): Number of samples to draw.

        Returns:
            SamplingRecord: Sampling record containing information about the
                sampled points.
        """
        points = torch.empty([n_samples, self.dim], device=self.min.device).uniform_(
            generator=self.generator
        )
        points = self.min[None, :] + self.extent[None, :] * points
        samples = SamplingRecord(point=points, pdf=self.pdf)
        return samples


def sample_sphere(n_samples: int, device: torch.device | str = "cuda") -> torch.Tensor:
    # TODO: Fix efficiency of uniform sphere sampling.
    # TODO: Add generator to make experiments reproducible.
    sphere_samples = torch.randn([n_samples, 3], device=device)
    sphere_samples_norm = torch.linalg.vector_norm(
        sphere_samples, axis=-1, keepdim=True
    )
    sphere_samples = sphere_samples / sphere_samples_norm
    return sphere_samples


def sample_circle(n_samples: int, device: torch.device | str = "cuda") -> torch.Tensor:
    # TODO: Add generator to make experiments reproducible.
    samples = torch.empty([n_samples], device=device).uniform_(-torch.pi, torch.pi)
    samples = torch.stack([samples.cos(), samples.sin()], -1)
    return samples