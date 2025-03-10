# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

from argparse import ArgumentParser

import torch

from iskra.directional import (
    face_tangent_bundle,
    to_extrinsic_n_rosy,
    to_intrinsic_n_rosy,
    transport_from_face,
)
from iskra.io import load
from iskra.mesh import Mesh
from iskra.topology import edge_flaps

if __name__ == "__main__":
    parser = ArgumentParser(description="Compute and visualize a smooth N-RoSy field.")
    parser.add_argument("mesh_path", type=str, help="The path of the mesh to load.")
    args = parser.parse_args()

    mesh = Mesh.from_path(args.mesh_path, device="cpu")
    mesh.geom.normalize()
    faces, verts = mesh.topo.faces, mesh.geom.vertices

    flaps = edge_flaps(faces)
    tangents, binormals, connection = face_tangent_bundle(verts, faces, flaps)

    source = 0
    n = 4
    intrinsic = to_intrinsic_n_rosy(
        torch.tensor([1.0, 0.0, 0]), tangents[0], binormals[0], n
    )
    transported = transport_from_face(
        source, intrinsic, faces.shape[0], flaps, connection, n
    )
    extrinsic = to_extrinsic_n_rosy(transported, tangents, binormals, n)

    try:
        import polyscope as ps

        ps.init()
        ps_mesh = ps.register_surface_mesh("mesh", verts.numpy(), faces.numpy())

        for i in range(n):
            ps_mesh.add_vector_quantity(
                f"extrinsic_{i}",
                extrinsic[:, i, :],
                radius=0.001,
                length=0.002,
                color=(0.9, 0.1, 0.1),
                defined_on="faces",
                enabled=True,
            )
        ps.show()
    except ImportError:
        print(
            "Could not import Polyscope to visualize the results."
            "Install it by running: pip install polyscope"
        )
