# Copyright (c) 2022 - present, Ana Dodik. All rights reserved.

import json
from dataclasses import dataclass, field, fields
from io import TextIOBase
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Type, cast

import fsspec
import igl
import meshio
import torch
from fsspec.implementations.cached import WholeFileCacheFileSystem
from fsspec.implementations.github import GithubFileSystem

from iskra.io.io_ext import (  # pyright: ignore
    ParsedMesh,
    load_obj_file,
    load_obj_string,
)
from iskra.logging.logging import getLogger

try:
    import trimesh
except ModuleNotFoundError:
    _trimesh_installed = False
else:
    _trimesh_installed = True

try:
    import igl
except ModuleNotFoundError:
    _libigl_installed = False
else:
    _libigl_installed = True

LOGGER = getLogger(__name__)
LOGGER.setLevel("INFO")


class OdedFileSystem(GithubFileSystem):  # type: ignore
    protocol = "oded"

    def __init__(self, sha: Optional[str] = None, **kwargs: Any):
        super().__init__("odedstein", "meshes", **kwargs)


class LibiglFileSystem(GithubFileSystem):  # type: ignore
    protocol = "libigl"

    def __init__(self, sha: Optional[str] = None, **kwargs: Any):
        super().__init__("libigl", "libigl-tutorial-data", **kwargs)


def _wrap_cached(cls: Type[Any]) -> Type[Any]:
    class CachedFileSystem(WholeFileCacheFileSystem):  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any):
            target_options = kwargs.pop("target_options", None)
            kwargs.pop("target_protocol", None)
            if target_options is not None:
                fs = cls(**target_options)
            else:
                fs = cls()
            super().__init__(*args, fs=fs, **kwargs)

    return CachedFileSystem


if "oded" not in fsspec.registry:
    fsspec.register_implementation("oded", _wrap_cached(OdedFileSystem))
if "libigl" not in fsspec.registry:
    fsspec.register_implementation("libigl", _wrap_cached(LibiglFileSystem))


@dataclass(kw_only=True, slots=True)
class MeshData:
    """Contains parsed mesh data.

    Attributes:
        tetrahedra (torch.Tensor): `[T, 4]` tensor of tetrahedra indices.
        triangles (torch.Tensor): `[F, 3]` tensor of triangle indices.
        lines (torch.Tensor): `[E, 2]` tensor of line segment indices.
        positions (torch.Tensor): `[V, 2 | 3]` tensor of vertex positions.
        uvs (torch.Tensor): `[U, 2]` tensor of uv coordinates.
        uvs_idx (torch.Tensor): `[F, 3]` tensor of uv coordinates indices.
        normals (torch.Tensor): `[N, 2 | 3]` tensor of normal vectors.
        normals_idx (torch.Tensor): `[F, 3]` tensor of normal vector indices.
        material_ids (torch.Tensor): `[M, 1]` tensor of per-face material IDs.
    """

    positions: torch.Tensor = field(
        default_factory=lambda: torch.empty([0, 3], dtype=torch.float32)
    )
    tetrahedra: torch.Tensor = field(
        default_factory=lambda: torch.empty([0, 4], dtype=torch.long)
    )
    triangles: torch.Tensor = field(
        default_factory=lambda: torch.empty([0, 3], dtype=torch.long)
    )
    lines: torch.Tensor = field(
        default_factory=lambda: torch.empty([0, 2], dtype=torch.long)
    )
    uvs: torch.Tensor = field(
        default_factory=lambda: torch.empty([0, 2], dtype=torch.float32)
    )
    uvs_idx: torch.Tensor = field(
        default_factory=lambda: torch.empty([0, 3], dtype=torch.long)
    )
    normals: torch.Tensor = field(
        default_factory=lambda: torch.empty([0, 3], dtype=torch.float32)
    )
    normals_idx: torch.Tensor = field(
        default_factory=lambda: torch.empty([0, 3], dtype=torch.long)
    )
    material_ids: torch.Tensor = field(
        default_factory=lambda: torch.empty([0, 3], dtype=torch.long)
    )

    def to(self, fdtype: torch.dtype, device: torch.device | str) -> "MeshData":
        tensor_dict = {}
        for field in fields(self):  # noqa: F402
            tensor = getattr(self, field.name)
            if torch.is_floating_point(tensor):
                tensor = tensor.to(dtype=fdtype, device=device)
            else:
                tensor = tensor.to(device=device)
            tensor_dict[field.name] = tensor
        return MeshData(**tensor_dict)


def load_obj(
    file: str | Path | TextIOBase,
    fdtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> MeshData:
    if isinstance(file, TextIOBase):
        parsed = load_obj_string(file.read())
    elif isinstance(file, str):
        parsed = load_obj_file(file)
    elif isinstance(file, Path):
        parsed = load_obj_file(str(file))
    else:
        raise TypeError(f"Argument file with type {type(file)} is not supported")
    LOGGER.info(parsed)
    return MeshData(
        positions=parsed.positions(),
        normals=parsed.normals(),
        normals_idx=parsed.normal_idx(),
        uvs=parsed.texcoords(),
        uvs_idx=parsed.texcoord_idx(),
        material_ids=parsed.material_ids(),
        triangles=parsed.faces(),
        lines=parsed.lines(),
    ).to(fdtype=fdtype, device=device)


def _load_trimesh(
    path: str, fdtype: torch.dtype = torch.float32, device: torch.device | str = "cpu"
) -> MeshData:
    suffix = Path(path).suffix
    with fsspec.open(path) as mesh_file:
        trimesh_mesh = trimesh.load_mesh(  # pyright: ignore
            mesh_file,
            maintain_order=True,
            merge_tex=True,
            merge_norm=True,
            file_type=suffix,
        )
        LOGGER.info(f"Loaded mesh with {trimesh_mesh.vertices.shape[0]} vertices.")
        faces = torch.tensor(trimesh_mesh.faces, dtype=torch.long)
        vertices = torch.tensor(trimesh_mesh.vertices, dtype=torch.float32)
        if vertices.shape[1] == 3 and torch.all(vertices[:, 2] == 0.0):
            vertices = vertices[:, :2]
        uvs = None
        if isinstance(trimesh_mesh.visual, trimesh.visual.ColorVisuals):  # pyright: ignore
            LOGGER.warning(f"Could not find texture for mesh {path}!")
            uvs = torch.zeros_like(vertices[:, :2])
        elif isinstance(trimesh_mesh.visual, trimesh.visual.TextureVisuals):  # pyright: ignore
            uvs = torch.tensor(
                trimesh_mesh.visual.uv,  # pyright:ignore
                device=device,
                dtype=torch.float32,
            )
        assert uvs is not None
        return MeshData(triangles=faces, positions=vertices, uvs=uvs).to(
            fdtype=fdtype, device=device
        )


def _load_meshio(
    path: str, fdtype: torch.dtype = torch.float32, device: torch.device | str = "cpu"
) -> MeshData:
    suffix = Path(path).suffix
    with fsspec.open(path, mode="r") as mesh_file:
        meshio_mesh = meshio.read(
            mesh_file, file_format=meshio.extension_to_filetypes[suffix][0]
        )
        tet_cells = [cells for cells in meshio_mesh.cells if cells.type == "tetra"]
        if len(tet_cells) != 1:
            raise ValueError(
                f"Found {len(tet_cells)} sets of tetrahedra in file {path}."
            )
        vertices = torch.tensor(meshio_mesh.points, dtype=torch.float32, device=device)
        faces = torch.tensor(tet_cells[0].data, dtype=torch.long, device=device)
        uvs = torch.zeros_like(vertices)
    return MeshData(triangles=faces, positions=vertices, uvs=uvs).to(
        fdtype=fdtype, device=device
    )


def _load_libigl(
    path: str, fdtype: torch.dtype = torch.float32, device: torch.device | str = "cpu"
) -> MeshData:
    LOGGER.warning("Falling back to libigl.")
    vertices, faces = igl.read_triangle_mesh(str(path))
    vertices = torch.tensor(vertices, device=device, dtype=torch.float32)
    triangles = torch.tensor(faces, device=device, dtype=torch.long)
    return MeshData(triangles=triangles, positions=vertices).to(
        fdtype=fdtype, device=device
    )


def load(
    path: str | Path,
    fdtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> MeshData:
    suffix = Path(path).suffix
    if suffix == ".msh" or suffix == ".mesh":
        return _load_meshio(str(path), fdtype=fdtype, device=device)
    elif suffix == ".obj":
        with fsspec.open(path, mode="r") as file:
            if TYPE_CHECKING:
                file = cast(TextIOBase, file)
            return load_obj(file, fdtype=fdtype, device=device)
    elif _libigl_installed:
        return _load_libigl(str(path), fdtype=fdtype, device=device)
    elif _trimesh_installed:
        return _load_trimesh(str(path), fdtype=fdtype, device=device)
    else:
        raise RuntimeError(f"Failed to load file: {path}.")
