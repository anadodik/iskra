# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

from argparse import ArgumentParser

import numpy as np
import torch

from iskra.dec import laplacian
from iskra.geometry import triangle_areas
from iskra.mesh import Mesh
from iskra.parameterization import (
    boundary_matrix,
    conformal_laplacian,
    scp_solve,
    symmetric_dirichlet_energy,
    triangle_to_local,
    uv_local,
)
from iskra.sparse_linalg import default_solver
from iskra.topology import face_index, get_subfaces

if __name__ == "__main__":
    parser = ArgumentParser(description="Demonstrates an inverse SCP parameterization.")
    parser.add_argument("mesh_path", type=str, help="The path of the mesh to load.")
    args = parser.parse_args()

    dtype = torch.double
    device = "cpu"
    mesh, _ = Mesh.from_path(args.mesh_path, dtype=dtype, device=device)
    mesh.geom.normalize()
    faces, verts = mesh.topo.faces, mesh.geom.vertices

    edges, face_edges, face_edge_sign = get_subfaces(faces)

    boundary_mat = boundary_matrix(mesh.n_vertices, faces, dtype=dtype)

    def compute_scp(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        conformal_lap = conformal_laplacian(verts, faces, clamp_min=1e-8)
        return scp_solve(conformal_lap, boundary_mat)

    rest_local = triangle_to_local(verts, faces)
    rest_areas = triangle_areas(face_index(verts, faces))

    verts_opt = torch.nn.Parameter(verts.clone())
    lr = 0.01
    optimizer = torch.optim.SGD([verts_opt], lr=lr)
    lap, mass = laplacian(verts, faces, clamp_min=0.0)
    h1_solver = default_solver(mass + 0.8 * lap)
    uv_opt = compute_scp(verts_opt, faces)

    def step_fn():
        uv_opt = compute_scp(verts_opt, faces)
        param_local = uv_local(uv_opt, faces)
        energy = symmetric_dirichlet_energy(rest_local, param_local, rest_areas)
        energy.mean().backward()
        print(energy.mean())
        with torch.no_grad():
            if verts_opt.grad is None:
                raise RuntimeError("verts_var.grad is None!")
            if not torch.isfinite(verts_opt.grad).all():
                raise RuntimeError("verts_var.grad not finite!")
            verts_opt.grad = h1_solver(mass @ verts_opt.grad)
            verts_opt.grad -= verts_opt.grad.mean(0, keepdim=True)
            print(verts_opt.grad.min(), verts_opt.grad.max())

            # energy_new = symmetric_dirichlet(
            #     rest_local,
            #     uv_local(compute_scp(verts_opt - lr * verts_opt.grad, faces), faces),
            # )
            # n_shrinks = 0
            # while energy_new.mean() > energy.mean():
            #     verts_opt.grad *= 0.1
            #     energy_new = symmetric_dirichlet(
            #         rest_local,
            #         uv_local(
            #             compute_scp(verts_opt - lr * verts_opt.grad, faces), faces
            #         ),
            #     )
            #     n_shrinks += 1
            # if n_shrinks > 0:
            #     print(f"Shrunk the learning rate {n_shrinks} times.")

        return energy

    with torch.no_grad():
        uv_opt = compute_scp(verts_opt, faces)
        param_local = uv_local(uv_opt, faces)
        # energy = symmetric_dirichlet(rest_local, param_local, rest_areas)
    try:
        import polyscope as ps

        ps.init()
        ps_mesh_init = ps.register_surface_mesh(
            "Mesh SCP", verts.numpy() + np.array([1.0, 0.0, 0.0]), faces.numpy()
        )
        print(uv_opt.shape, verts.shape)
        ps_mesh_init.add_parameterization_quantity(
            "Param SCP", uv_opt.detach().numpy(), enabled=True
        )
        ps_mesh = ps.register_surface_mesh("Mesh", verts.numpy(), faces.numpy())
        ps_mesh.add_scalar_quantity(
            "det(J)",
            torch.linalg.det(rest_local).numpy(),
            defined_on="faces",
            enabled=True,
        )
        ps_mesh.add_parameterization_quantity(
            "Param SymmDir", uv_opt.detach().numpy(), enabled=True
        )
        ps_edges = ps.register_curve_network(
            "edges", verts.numpy(), edges.numpy(), enabled=False, radius=0.01
        )

        ps_param_mesh = ps.register_surface_mesh(
            "UV Mesh", uv_opt.detach().numpy(), faces.numpy(), edge_width=1
        )
        # ps_param_mesh.add_scalar_quantity(
        #     "energy", energy.detach().numpy(), defined_on="faces", enabled=True
        # )
        ve = face_index(verts_opt, edges).detach()
        ps_edges.add_scalar_quantity(
            "metric",
            torch.linalg.vector_norm(ve[..., 1, :] - ve[..., 0, :], dim=-1).numpy(),
            defined_on="edges",
        )

        optimizing = False

        def callback():
            global optimizing

            if ps.imgui.Button("Step" if not optimizing else "Step"):
                #     optimizing = not optimizing
                # if optimizing:
                for _ in range(1):
                    optimizer.zero_grad()
                    energy = step_fn()
                    optimizer.step()
                    uv_opt = compute_scp(verts_opt, faces)
                    param_local = uv_local(uv_opt, faces)
                    energy = symmetric_dirichlet_energy(
                        rest_local, param_local, rest_areas
                    )

                    # ps_mesh.update_vertex_positions(verts_opt.detach().numpy())
                    ps_param_mesh.update_vertex_positions(uv_opt.detach().numpy())
                    ps_param_mesh.add_scalar_quantity(
                        "energy",
                        energy.detach().numpy(),
                        defined_on="faces",
                        enabled=True,
                    )
                    ps_mesh.add_parameterization_quantity(
                        "rand param", uv_opt.detach().numpy(), enabled=True
                    )
                    ve = face_index(verts_opt, edges).detach()
                    ps_edges.add_scalar_quantity(
                        "metric",
                        torch.linalg.vector_norm(
                            ve[..., 1, :] - ve[..., 0, :], dim=-1
                        ).numpy(),
                        defined_on="edges",
                    )

        ps.set_user_callback(callback)
        ps.show()
    except ImportError:
        print(
            "Could not import Polyscope to visualize the results."
            "Install it by running: pip install polyscope"
        )
