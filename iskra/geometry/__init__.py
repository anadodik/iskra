# Copyright (c) 2022 - present, Ana Dodik. All rights reserved.

import iskra.geometry.dual_quaternions as dual_quat
import iskra.geometry.quaternions as quat
from iskra.geometry.barycentric import (
    barycentric_coordinates,
    barycentric_interpolate,
    is_inside_pairwise,
    tetrahedron_barycentric_coordinates,
    triangle_barycentric_coordinates,
)
from iskra.geometry.bbox import BBox
from iskra.geometry.coordinate_system import (
    normal_coordinate_system,
    triangle_coordinate_system,
)
from iskra.geometry.cotan_weights import cotan_weights, cotan_weights_intrinsic
from iskra.geometry.distances import (
    closest_edge,
    closest_triangle,
    edge_project,
    point_dist,
    point_dist_matrix,
    point_edge_dist,
    point_edge_dist_matrix,
    point_simplex_dist,
    point_simplex_dist_matrix,
    point_tetrahedron_dist,
    point_tetrahedron_dist_matrix,
    point_triangle_dist,
    point_triangle_dist_matrix,
    simplex_project,
    tetrahedron_project,
    triangle_project,
)
from iskra.geometry.dual_quaternions import DualQuaternion
from iskra.geometry.element_quality import abs_tetrahedron_heights, triangle_altitudes
from iskra.geometry.extrude_boundary import extrude_boundary_polygon
from iskra.geometry.normals import edge_normals, triangle_area_normals, triangle_normals
from iskra.geometry.quaternions import Quaternion
from iskra.geometry.volume import (
    edge_lengths,
    tetrahedron_volumes,
    tetrahedron_volumes_intrinsic,
    triangle_areas,
    triangle_areas_intrinsic,
    volume_form,
)

__all__ = [
    "point_dist",
    "point_edge_dist",
    "point_triangle_dist",
    "point_tetrahedron_dist",
    "point_dist_matrix",
    "point_edge_dist_matrix",
    "point_triangle_dist_matrix",
    "point_tetrahedron_dist_matrix",
    "point_simplex_dist",
    "point_simplex_dist_matrix",
    "closest_edge",
    "closest_triangle",
    "triangle_altitudes",
    "abs_tetrahedron_heights",
    "edge_lengths",
    "triangle_areas",
    "tetrahedron_volumes",
    "volume_form",
    "triangle_barycentric_coordinates",
    "tetrahedron_barycentric_coordinates",
    "barycentric_coordinates",
    "barycentric_interpolate",
    "is_inside_pairwise",
    "triangle_area_normals",
    "triangle_normals",
    "edge_length_normals",
    "edge_normals",
    "edge_projection_pairwise",
    "edge_project",
    "triangle_project",
    "tetrahedron_project",
    "simplex_project",
    "quat",
    "Quaternion",
    "dual_quat",
    "DualQuaternion",
    "extrude_boundary_polygon",
    "BBox",
    "normal_coordinate_system",
    "triangle_coordinate_system",
    "cotan_weights",
    "cotan_weights_intrinsic",
    "triangle_areas_intrinsic",
    "tetrahedron_volumes_intrinsic",
]
