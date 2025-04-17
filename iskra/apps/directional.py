# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

from argparse import ArgumentParser

import torch

from iskra.directional import (
    face_connection_laplacian,
    face_tangent_bundle,
    to_extrinsic_n_rosy,
    to_intrinsic_n_rosy,
)
from iskra.mesh import Mesh
from iskra.sparse import min_quadratic_energy
from iskra.topology import edge_flaps

if __name__ == "__main__":
    parser = ArgumentParser(description="Compute and visualize a smooth N-RoSy field.")
    parser.add_argument("mesh_path", type=str, help="The path of the mesh to load.")
    parser.add_argument("--n", type=int, default=4, help="Degree of N-RoSy field.")
    args = parser.parse_args()

    source = 0
    n = args.n

    mesh = Mesh.from_path(args.mesh_path, device="cpu")
    mesh.geom.normalize()
    faces, verts = mesh.topo.faces, mesh.geom.vertices

    flaps = edge_flaps(faces)
    tangents, binormals, connection, _ = face_tangent_bundle(verts, faces, flaps)
    connection_n = connection**n

    t_source, b_source = tangents[source : source + 1], binormals[source : source + 1]
    source_v = -1.0 * t_source + 0.0 * b_source
    intrinsic = to_intrinsic_n_rosy(source_v, t_source, b_source, n)

    laplacian = face_connection_laplacian(verts, faces, flaps, connection_n)

    transported = min_quadratic_energy(
        laplacian,
        torch.zeros([faces.shape[0]], dtype=torch.cfloat),
        torch.tensor([source]),
        intrinsic,
    )
    extrinsic = to_extrinsic_n_rosy(transported, tangents, binormals, n)

    try:
        import polyscope as ps

        ps.init()
        ps_mesh = ps.register_surface_mesh("mesh", verts.numpy(), faces.numpy())

        source_vs = extrinsic.clone()
        source_vs[:source, ...] = 0.0
        source_vs[source + 1 :, ...] = 0.0
        for i in range(n):
            ps_mesh.add_vector_quantity(
                f"source_{i}",
                source_vs[:, i, :],
                radius=0.002,
                length=0.015,
                color=(0.1, 0.9, 0.1),
                defined_on="faces",
                enabled=True,
            )

        for i in range(n):
            ps_mesh.add_vector_quantity(
                f"extrinsic_{i}",
                extrinsic[:, i, :],
                radius=0.002,
                length=0.01,
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
