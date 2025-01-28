# Copyright (c) 2022 - present, Ana Dodik. All rights reserved.

import json
from io import TextIOBase
from pathlib import Path
from typing import Any, Optional, Type

import fsspec
import igl
import meshio
import requests  # type: ignore
import torch
import trimesh
from fsspec.implementations.cached import WholeFileCacheFileSystem
from fsspec.implementations.github import GithubFileSystem
from fsspec.implementations.http import HTTPFile
from fsspec.implementations.memory import MemoryFile

from iskra.io.io_ext import ParsedOBJ, load_obj_file, load_obj_string
from iskra.logging.logging import getLogger

LOGGER = getLogger(__name__)
LOGGER.setLevel("INFO")


class AnaFileSystem(GithubFileSystem):  # type: ignore
    protocol = "ana"
    _TOKEN = (
        "github_pat_11ABGRQQA0xtA5wprWWLTl_"
        "pSjPKFefHgds9SbYGYwHgkqL89l0Kz6BEhUZmq0Xz3nVHEVADGZGamfyswC"
    )
    _CONTENT_URL = "https://api.github.com/repos/{org}/{repo}/contents/{path}"

    def __init__(self, sha: Optional[str] = None, **kwargs: Any):
        super().__init__(
            "anadodik",
            "geometry-data",
            username="anadodik",
            token=self._TOKEN,
            **kwargs,
        )

    def _open(  # type: ignore
        self,
        path,
        mode="rb",
        block_size=None,
        autocommit=True,
        cache_options=None,
        sha=None,
        **kwargs,
    ) -> HTTPFile:
        if mode not in ["rb", "r"]:
            raise NotImplementedError
        path = self._strip_protocol(path)
        url = self._CONTENT_URL.format(org=self.org, repo=self.repo, path=path)
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": self._TOKEN,
            "X-GitHub-Api-Version": "2022-11-28",
        }
        metadata_req = requests.get(url, headers=headers, **AnaFileSystem().kw)
        metadata = json.loads(metadata_req.text)
        if metadata["type"] != "file":
            raise FileNotFoundError(f"Cannot open path with type {metadata['type']}")
        data = requests.get(metadata["download_url"]).content
        return MemoryFile(None, None, data)


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
if "ana" not in fsspec.registry:
    fsspec.register_implementation("ana", _wrap_cached(AnaFileSystem))
if "libigl" not in fsspec.registry:
    fsspec.register_implementation("libigl", _wrap_cached(LibiglFileSystem))


def _load_trimesh(
    path: str, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    suffix = Path(path).suffix
    with fsspec.open(path) as mesh_file:
        trimesh_mesh = trimesh.load_mesh(
            mesh_file,
            maintain_order=True,
            merge_tex=True,
            merge_norm=True,
            file_type=suffix,
        )
        LOGGER.info(f"Loaded mesh with {trimesh_mesh.vertices.shape[0]} vertices.")
        faces = torch.tensor(trimesh_mesh.faces, device=device, dtype=torch.int64)
        vertices = torch.tensor(
            trimesh_mesh.vertices, device=device, dtype=torch.float32
        )
        if vertices.shape[1] == 3 and torch.all(vertices[:, 2] == 0.0):
            vertices = vertices[:, :2]
        uvs = None
        if isinstance(trimesh_mesh.visual, trimesh.visual.ColorVisuals):
            LOGGER.warning(f"Could not find texture for mesh {path}!")
            uvs = torch.zeros_like(vertices[:, :2])
        elif isinstance(trimesh_mesh.visual, trimesh.visual.TextureVisuals):
            uvs = torch.tensor(
                trimesh_mesh.visual.uv, device=device, dtype=torch.float32
            )
        assert uvs is not None
        return faces, vertices, uvs, None, None


def _load_meshio(
    path: str, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    return faces, vertices, uvs, None, None


def load_obj(
    file: str | Path | TextIOBase, device: torch.device | str = "cpu"
) -> ParsedOBJ:
    if isinstance(file, TextIOBase):
        parsed = load_obj_string(file.read())
    elif isinstance(file, str):
        parsed = load_obj_file(file)
    elif isinstance(file, Path):
        parsed = load_obj_file(str(file))
    else:
        raise TypeError(f"Argument file with type {type(file)} is not supported")
    return parsed


def _load_io_ext(
    path: str, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with fsspec.open(path, "r") as mesh_file:
        try:
            parsed = load_obj(mesh_file)
            vertices = torch.tensor(
                parsed.positions, device=device, dtype=torch.float32
            )
            triangles = torch.tensor(parsed.triangles, device=device, dtype=torch.long)
            lines = torch.tensor(parsed.lines, device=device, dtype=torch.long)
            normals = torch.tensor(parsed.normals, device=device, dtype=torch.float32)
            material_ids = torch.tensor(
                parsed.material_ids, device=device, dtype=torch.long
            )
        except Exception as e:
            print(e)
            vertices, faces = igl.read_triangle_mesh(str(path))
            vertices = torch.tensor(vertices, device=device, dtype=torch.float32)
            triangles = torch.tensor(faces, device=device, dtype=torch.long)
            lines = torch.tensor([], device=device, dtype=torch.long)
        if triangles.shape[0] > 0 and lines.shape[0] > 0:
            raise ValueError(
                "Cannot create Mesh object from file data: "
                "OBJ file contains both triangles and lines."
            )
        faces: torch.Tensor
        if triangles.shape[0] > 0:
            faces = triangles
        elif lines.shape[0] > 0:
            faces = lines
        else:
            faces = torch.arange(vertices.shape[0], device=device, dtype=torch.long)
            faces = faces[:, None]
        uvs = torch.zeros_like(vertices)[:, :2]
        return faces, vertices, uvs, normals, material_ids


def load(
    path: str, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    suffix = Path(path).suffix
    if suffix == ".msh" or suffix == ".mesh":
        return _load_meshio(path, device=device)
    elif suffix == ".obj":
        return _load_io_ext(path, device=device)
    else:
        return _load_trimesh(path, device=device)
