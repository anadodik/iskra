# Copyright (c) 2025 - present, Ana Dodik. All rights reserved.

from argparse import ArgumentParser

import torch

from iskra.dec import laplacian
from iskra.geometry import triangle_areas, triangle_coordinate_system
from iskra.mesh import Mesh
from iskra.sparse_linalg import CholeskySolver, min_quadratic_energy
from iskra.topology import boundary, face_index, get_subfaces, ordered_boundary_edges


def triangle_to_local(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    triangles = face_index(verts, faces)
    _, t, b = triangle_coordinate_system(triangles)
    edge_vecs = triangles[..., 1:, :] - triangles[..., 0:1, :]
    world_to_local = torch.stack([t, b], -2)
    local = world_to_local @ edge_vecs.mT
    return local


def uv_local(uv: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    # Do not project on a local coordinate frame because that will
    # leave us not knowing if there is a flip or not!
    triangles = face_index(uv, faces)
    edge_vecs = triangles[..., 1:, :] - triangles[..., 0:1, :]
    return edge_vecs.mT


def symmetric_dirichlet(
    rest_local: torch.Tensor, param_local: torch.Tensor, rest_areas: torch.Tensor
) -> torch.Tensor:
    # TODO: Can you do HVP? Replace vmap with vertmap?
    jac = param_local @ torch.linalg.inv(rest_local)
    energy_fwd = (jac**2).sum((-2, -1))
    energy_bwd = (torch.linalg.inv(jac) ** 2).sum((-2, -1))
    energy = rest_areas * (energy_fwd + energy_bwd)

    is_flipped = torch.linalg.det(param_local.mT) <= 0
    energy = torch.where(is_flipped, float("inf"), energy)
    return energy


symmetric_dirichlet = torch.compile(
    torch.vmap(symmetric_dirichlet, (0, 0, 0)), fullgraph=True, dynamic=True
)


# def symmetric_dirichlet_2(
#     rest_local: torch.Tensor, param_triangles: torch.Tensor, rest_areas: torch.Tensor
# ) -> torch.Tensor:
#     edge_vecs = param_triangles[..., 1:, :] - param_triangles[..., 0:1, :]
#     param_local = edge_vecs.mT
#     jac = param_local @ torch.linalg.inv(rest_local)
#     energy_fwd = (jac**2).sum((-2, -1))
#     energy_bwd = (torch.linalg.inv(jac) ** 2).sum((-2, -1))
#     energy = rest_areas * (energy_fwd + energy_bwd)
#
#     is_flipped = torch.linalg.det(param_local.mT) <= 0
#     energy = torch.where(is_flipped, float("inf"), energy)
#     return energy[..., None]
#
#
# sd_jac = torch.func.vmap(
#     torch.func.hessian(symmetric_dirichlet_2, 1), in_dims=(0, 0, 0)
# )

if __name__ == "__main__":
    # TODO: Rename file to AQP.
    parser = ArgumentParser(description="Demonstrates a SLIM parameterization.")
    parser.add_argument("mesh_path", type=str, help="The path of the mesh to load.")
    args = parser.parse_args()

    dtype = torch.double
    device = "cpu"
    mesh, _ = Mesh.from_path(args.mesh_path, fdtype=dtype, device=device)
    mesh.geom.normalize()
    faces, verts = mesh.topo.faces, mesh.geom.vertices

    # Assume one boundary loop, and take first vertex of each edge:
    bdr = ordered_boundary_edges(boundary(faces))[0][:, 0]
    n_bdr_verts = bdr.shape[0]
    t = torch.linspace(
        0, 2 * torch.pi - (2 * torch.pi) / n_bdr_verts, n_bdr_verts, dtype=dtype
    )
    bdr_uv = torch.stack([torch.cos(t), torch.sin(t)], -1) + 0.5

    edges, face_edges, face_edge_sign = get_subfaces(faces)
    rest_local = triangle_to_local(verts, faces)
    rest_areas = triangle_areas(face_index(verts, faces))

    lap, mass = laplacian(verts, faces, clamp_min=0.0)
    rhs = torch.zeros([verts.shape[0], 2], dtype=dtype, device=device)
    uv_init = min_quadratic_energy(lap, rhs, bdr, bdr_uv)
    # # print(torch.sort(triangle_areas(face_index(uv_init, faces))))

    # param_local = uv_local(uv_init, faces)
    # jac = sd_jac(rest_local, face_index(uv_init, faces), rest_areas)
    # idx_i = torch.cat([faces.flatten(), faces.flatten() + mesh.topo.n_faces])
    # idx_j = torch.cat([faces.flatten(), faces.flatten() + mesh.topo.n_faces])
    # print(jac.permute(0, 1, 2, 4, 3, 5).shape)
    # print(idx_i.shape)
    # # vjp_fn = torch.func.vjp(lambda x: uv_local(x, faces), uv_init)[1]
    # # print(vjp_fn(jac)[0].shape)
    # quit()

    uv_opt = torch.nn.Parameter(uv_init)
    lr = 100
    optimizer = torch.optim.SGD([uv_opt], lr=lr)
    h1_solver = CholeskySolver(mass + 0.5 * lap)

    param_local = uv_local(uv_opt, faces)
    energy = symmetric_dirichlet(rest_local, param_local, rest_areas)
    quit()

    def step_fn():
        param_local = uv_local(uv_opt, faces)
        energy = symmetric_dirichlet(rest_local, param_local, rest_areas)
        energy.mean().backward()
        with torch.no_grad():
            if uv_opt.grad is None:
                raise RuntimeError("verts_var.grad is None!")
            if not torch.isfinite(uv_opt.grad).all():
                raise RuntimeError("verts_var.grad not finite!")
            grad = h1_solver(mass @ uv_opt.grad)
            grad -= grad.mean(0, keepdim=True)

            energy_new = symmetric_dirichlet(
                rest_local, uv_local(uv_opt - lr * uv_opt.grad, faces), rest_areas
            )
            n_shrinks = 0
            while energy_new.mean() > energy.mean():
                grad *= 0.1
                energy_new = symmetric_dirichlet(
                    rest_local, uv_local(uv_opt - lr * grad, faces), rest_areas
                )
                n_shrinks += 1
            if n_shrinks > 0:
                print(f"Shrunk the learning rate {n_shrinks} times.")
            uv_opt.grad = grad

        return energy

    energy = step_fn()
    try:
        import polyscope as ps

        ps.init()
        ps_mesh = ps.register_surface_mesh("mesh", verts.numpy(), faces.numpy())
        ps_mesh.add_scalar_quantity(
            "det",
            torch.linalg.det(rest_local).numpy(),
            defined_on="faces",
            enabled=True,
        )
        ps_mesh.add_scalar_quantity(
            "face_area", mesh.geom.face_areas.numpy(), defined_on="faces"
        )
        ps_mesh.add_parameterization_quantity(
            "rand param", uv_opt.detach().numpy(), enabled=True
        )

        ps_boundary = ps.register_curve_network(
            "boundary",
            face_index(verts, bdr).numpy(),
            edges="loop",
        )

        ps_param_mesh = ps.register_surface_mesh(
            "param_mesh", uv_opt.detach().numpy(), faces.numpy(), edge_width=1
        )
        ps_param_boundary = ps.register_curve_network(
            "param_boundary", bdr_uv.numpy(), edges="loop"
        )
        ps_param_mesh.add_scalar_quantity(
            "energy",
            energy.detach().numpy(),
            defined_on="faces",
            enabled=True,
        )

        optimizing = False

        def callback():
            global optimizing

            if ps.imgui.Button(
                "Start Optimization" if not optimizing else "Stop Optimizing"
            ):
                optimizing = not optimizing
            if optimizing:
                for _ in range(10):
                    optimizer.zero_grad()
                    energy = step_fn()
                    optimizer.step(lambda: step_fn().mean())

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

        ps.set_user_callback(callback)
        ps.show()
    except ImportError:
        print(
            "Could not import Polyscope to visualize the results."
            "Install it by running: pip install polyscope"
        )
