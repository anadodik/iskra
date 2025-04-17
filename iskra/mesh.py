# Copyright (c) 2022 - present, Ana Dodik. All rights reserved.

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Self

import torch

from iskra.geometry import BBox, triangle_area_normals, triangle_areas
from iskra.geometry.normals import edge_length_normals
from iskra.geometry.volume import edge_lengths, tetrahedron_volumes, volume_form
from iskra.io import load
from iskra.io.io import MeshData
from iskra.logging.logging import getLogger
from iskra.topology import boundary, face_index, get_subfaces, reduce_on_subface

LOGGER = getLogger(__name__)
LOGGER.setLevel("INFO")


class Topology(torch.nn.Module):
    def __init__(
        self,
        faces: torch.Tensor,
        vertices: torch.Tensor | int | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("faces", faces)
        if vertices is None:
            n_vertices = faces.max().item() + 1
            vertices = torch.arange(n_vertices, device=faces.device)
        elif isinstance(vertices, int):
            n_vertices = vertices
            vertices = torch.arange(n_vertices, device=faces.device)

        self.register_buffer("vertices", vertices)

        if TYPE_CHECKING:
            self.faces: torch.Tensor
            self.vertices: torch.Tensor

    @property
    def intrinsic_dim(self) -> int:
        return self.faces.shape[-1] - 1

    @property
    def n_faces(self) -> int:
        return self.faces.shape[0]

    @property
    def n_vertices(self) -> int:
        return self.vertices.shape[0]

    def subfaces(self, dim: int | None = -1) -> torch.Tensor:
        if dim is None:
            dim = self.intrinsic_dim
        elif dim < 0:
            dim = self.intrinsic_dim + dim

        if dim == self.intrinsic_dim:
            return self.faces
        elif dim > self.intrinsic_dim:
            return torch.zeros([0, dim + 1], device=self.faces.device, dtype=torch.long)

        subfaces = get_subfaces(self.faces, dim)
        subfaces = torch.flatten(subfaces, -3, -2)
        subfaces = torch.sort(subfaces, -1)[0]
        subfaces = torch.unique(subfaces, dim=-2)
        return subfaces

    def faces_to_subfaces(
        self, face_dim: int | None = None, subface_dim: int = -1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        faces: torch.Tensor = self.subfaces(face_dim)

        if subface_dim < 0:
            subface_dim = self.intrinsic_dim + subface_dim

        subfaces = get_subfaces(faces, subface_dim)
        n_subfaces = subfaces.shape[-2]
        subfaces = torch.flatten(subfaces, -3, -2)
        subfaces = torch.sort(subfaces, -1)[0]
        subfaces, face_to_subface = torch.unique(subfaces, dim=-2, return_inverse=True)
        return subfaces, face_to_subface.reshape(-1, n_subfaces)

    @cached_property
    def tetrahedra(self) -> torch.Tensor:
        return self.subfaces(3)

    @cached_property
    def triangles(self) -> torch.Tensor:
        return self.subfaces(2)

    @cached_property
    def edges(self) -> torch.Tensor:
        return self.subfaces(1)

    @cached_property
    def isolated_vertices(self) -> torch.Tensor:
        if self.intrinsic_dim == 0:
            return self.faces.flatten(-2, -1)
        mask = torch.ones(self.n_vertices, device=self.faces.device, dtype=torch.bool)
        mask[self.faces.unique()] = False  # type: ignore
        return self.vertices[mask]


class Geometry(torch.nn.Module):
    def __init__(self, topology: Topology, vertices: torch.Tensor) -> None:
        super().__init__()

        self.topo = topology

        self.register_buffer("vertices", vertices)
        self._vertex_normals = None
        if TYPE_CHECKING:
            self.vertices: torch.Tensor

    @property
    def ambient_dim(self) -> int:
        return self.vertices.shape[-1]

    @property
    def n_vertices(self) -> int:
        return self.vertices.shape[0]

    @property
    def bbox(self) -> BBox:
        return BBox.compute(self.vertices)

    def normalize(self, bbox: BBox | None = None) -> None:
        if bbox is not None:
            bbox = bbox.to(self.vertices.device)
        else:
            bbox = self.bbox
        max_extent = torch.max(bbox.extent)
        self.vertices = (self.vertices - bbox.min) / max_extent

    def __getitem__(self, faces: torch.Tensor) -> torch.Tensor:
        if isinstance(faces, torch.Tensor):
            return face_index(self.vertices, faces)
        else:
            return self.vertices[faces]

    @property
    def faces(self) -> torch.Tensor:
        return face_index(self.vertices, self.topo.faces)

    def subfaces(self, dim: int = -1) -> torch.Tensor:
        return face_index(self.vertices, self.topo.subfaces(dim))

    @property
    def tetrahedra(self) -> torch.Tensor:
        return self.subfaces(3)

    @property
    def triangles(self) -> torch.Tensor:
        return self.subfaces(2)

    @property
    def edges(self) -> torch.Tensor:
        return self.subfaces(1)

    @property
    def isolated_vertices(self) -> torch.Tensor:
        return face_index(self.vertices, self.topo.isolated_vertices)

    @property
    def area_face_normals(self) -> torch.Tensor:
        if self.topo.intrinsic_dim == 2 and self.ambient_dim in (2, 3):
            return triangle_area_normals(self.faces)
        elif self.topo.intrinsic_dim == 1 and self.ambient_dim == 2:
            return edge_length_normals(self.faces)
        else:
            raise NotImplementedError(
                f"Normals not implemented for "
                f"intrinsic_dim={self.topo.intrinsic_dim} "
                f"and ambient_dim={self.ambient_dim}"
            )

    @property
    def face_normals(self) -> torch.Tensor:
        return torch.nn.functional.normalize(self.area_face_normals, dim=-1)

    @property
    def vertex_normals(self) -> torch.Tensor:
        # normals = torch.
        # normals = face_to_subface_scatter_add(
        #     self.area_face_normals, self.topology.faces, self.geometry.n_vertices
        # )
        # return torch.nn.functional.normalize(normals, dim=-1)
        if self._vertex_normals is None:
            normals = reduce_on_subface(
                self.area_face_normals, self.topo.faces, self.vertices.shape[0], "sum"
            )
            return torch.nn.functional.normalize(normals, dim=-1)
        else:
            return self._vertex_normals

    @vertex_normals.setter
    def vertex_normals(self, value: torch.Tensor):
        self._vertex_normals = value

    @property
    def face_areas(self) -> torch.Tensor:
        return volume_form(self.faces)

    @property
    def vertex_areas(self) -> torch.Tensor:
        triple_area = reduce_on_subface(
            self.face_areas, self.topo.faces, self.topo.n_vertices, "sum"
        )
        return triple_area / 3


class Mesh(torch.nn.Module):
    def __init__(
        self,
        topology: torch.Tensor | Topology,
        geometry: torch.Tensor | Geometry,
    ) -> None:
        super().__init__()
        if isinstance(topology, torch.Tensor):
            topology = Topology(topology)
        if isinstance(geometry, torch.Tensor):
            geometry = Geometry(topology, geometry)
        self.topo: Topology = topology
        self.geom: Geometry = geometry

    def __iter__(self) -> Iterator[Topology | Geometry]:
        return iter([self.topo, self.geom])

    @property
    def vertices(self) -> torch.Tensor:
        return self.geom.vertices

    @property
    def faces(self) -> torch.Tensor:
        return self.topo.faces

    @property
    def intrinsic_dim(self) -> int:
        return self.topo.intrinsic_dim

    @property
    def ambient_dim(self) -> int:
        return self.geom.ambient_dim

    @property
    def n_vertices(self) -> int:
        return self.geom.n_vertices

    def deduplicate_vertices(
        self, vertex_values: list[torch.Tensor] | None = None
    ) -> "tuple[Mesh, torch.Tensor] | tuple[Mesh, torch.Tensor, list[torch.Tensor]]":
        # raise DeprecationWarning("Please sync with gemfields code.")
        vertices = self.geom.vertices
        vertices, unique_index = torch.unique(vertices, dim=0, return_inverse=True)
        faces = unique_index[self.topo.faces.flatten()].reshape(
            -1, self.topo.intrinsic_dim + 1
        )
        if vertex_values is not None:
            v_val_result_list = []
            for v_val in vertex_values:
                v_val_result = torch.zeros(
                    [vertices.shape[0], *v_val.shape[1:]],
                    dtype=v_val.dtype,
                    device=v_val.device,
                )
                v_val_result[unique_index] = v_val
                v_val_result_list.append(v_val_result)
            return Mesh(faces, vertices), unique_index, v_val_result_list

        return Mesh(faces, vertices), unique_index

    def boundary_mesh(self) -> "tuple[Mesh, torch.Tensor]":
        faces = boundary(self.topo.faces)
        device = faces.device

        vertex_idcs: torch.Tensor = faces.reshape(-1).unique()  # type: ignore
        vertices = face_index(self.geom.vertices, vertex_idcs)

        new_vertex_idcs = torch.arange(vertex_idcs.shape[0], device=device)
        inv_idx = torch.empty(self.n_vertices, dtype=torch.long, device=device)
        inv_idx[vertex_idcs] = new_vertex_idcs
        faces = inv_idx[faces.flatten()].reshape(-1, self.intrinsic_dim)
        # Previous two lines perform the following scatter-gather operation:
        #   inv_idx.scatter_(0, vertex_idcs, new_vertex_idcs)
        #   faces = torch.gather(inv_idx, 0, edges.flatten()).reshape(-1, 2)

        topology = Topology(faces)
        geometry = Geometry(topology, vertices)
        mesh = Mesh(topology, geometry)

        return mesh, vertex_idcs

    @classmethod
    def from_path(
        cls,
        path: Path | str,
        normalize: bool = False,
        fdtype: torch.dtype = torch.float32,
        device: str | torch.device = "cuda",
    ) -> tuple[Self, MeshData]:
        mesh_data = load(path, fdtype=fdtype, device=device)
        if mesh_data.triangles.shape[0] > 0 and mesh_data.lines.shape[0] > 0:
            raise ValueError(
                f"Cannot create Mesh object from file data at {path} ,"
                "file contains both triangles and lines."
            )
        faces: torch.Tensor
        if mesh_data.triangles.shape[0] > 0:
            faces = mesh_data.triangles
        elif mesh_data.lines.shape[0] > 0:
            faces = mesh_data.lines
        else:  # Point cloud
            faces = torch.arange(
                mesh_data.positions.shape[0], device=device, dtype=torch.long
            )[:, None]
        topology = Topology(faces, mesh_data.positions.shape[0])
        geometry = Geometry(topology, mesh_data.positions)

        mesh = cls(topology, geometry)
        if normalize:
            mesh.geom.normalize()
        return mesh, mesh_data
