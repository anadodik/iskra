// Copyright (c) 2022 - present, Ana Dodik. All rights reserved.

#include "rapidobj/rapidobj.hpp"
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/ndarray.h>
#include <nanobind/nanobind.h>

#include <iostream>
#include <sstream>

namespace nb = nanobind;

template <typename T, int D>
using Tensor =
    nb::ndarray<nb::pytorch, T, nb::shape<-1, D>>;

struct ParsedMesh
{
  // Cannot store Python objects directly,
  // as that would cause a reference counting issue.
  // Therefore, we store only the memory here.

  std::vector<int64_t> faces;
  std::vector<int64_t> texcoord_idx;
  std::vector<int64_t> normal_idx;

  size_t faces_shape[2] = {0, 3};

  std::vector<int64_t> lines;
  size_t lines_shape[2] = {0, 2};

  std::vector<int64_t> material_ids;
  size_t material_ids_shape[2] = {0, 1};

  std::vector<float> positions;
  size_t positions_shape[2] = {0, 3};

  std::vector<float> normals;
  size_t normals_shape[2] = {0, 3};

  std::vector<float> texcoords;
  size_t texcoords_shape[2] = {0, 2};

  std::string to_string() const
  {
    return std::string("ParsedMesh(\n") +
           "    positions=tensor(shape=[" + std::to_string(positions_shape[0]) + ", " + std::to_string(positions_shape[1]) + "]),\n" +
           "    normals=tensor(shape=[" + std::to_string(normals_shape[0]) + ", " + std::to_string(normals_shape[1]) + "]),\n" +
           "    texcoords=tensor(shape=[" + std::to_string(texcoords_shape[0]) + ", " + std::to_string(texcoords_shape[1]) + "]),\n" +
           "    lines=tensor(shape=[" + std::to_string(lines_shape[0]) + ", " + std::to_string(lines_shape[1]) + "]),\n" +
           "    faces=tensor(shape=[" + std::to_string(faces_shape[0]) + ", " + std::to_string(faces_shape[1]) + "]),\n" +
           "    material_ids=tensor(shape=[" + std::to_string(material_ids_shape[0]) + ", " + std::to_string(material_ids_shape[1]) + "]),\n" +
           ")";
  }
};

void parse_lines(const rapidobj::Lines &lines, ParsedMesh &mesh)
{
  size_t n_edges = 0;
  for (int nv : lines.num_line_vertices)
  {
    n_edges += nv - 1;
  }
  if (n_edges == 0)
  {
    return;
  }

  mesh.lines_shape[0] = n_edges;
  mesh.lines.resize(n_edges * 2);

  int v_i = 0;
  int e_i = 0;
  for (int nv : lines.num_line_vertices)
  {
    ++v_i;
    for (int i = 1; i < nv; ++i)
    {
      mesh.lines[e_i] = lines.indices[v_i - 1].position_index;
      mesh.lines[e_i + 1] = lines.indices[v_i].position_index;
      ++v_i;
      e_i += 2;
    }
  }
}

void parse_faces(const rapidobj::Mesh &rapidMesh, ParsedMesh &mesh)
{
  size_t n_faces = rapidMesh.num_face_vertices.size();
  if (n_faces == 0)
  {
    return;
  }
  size_t dim = rapidMesh.num_face_vertices[0];
  for (int nv : rapidMesh.num_face_vertices)
  {
    if (nv != dim || nv != dim)
    {
      throw std::runtime_error(
          "All faces in OBJ mesh must have the same number of vertices.");
    }
  }

  mesh.faces_shape[0] = n_faces;
  mesh.faces_shape[1] = dim;
  mesh.faces.resize(n_faces * dim);
  mesh.texcoord_idx.resize(n_faces * dim);
  mesh.normal_idx.resize(n_faces * dim);

  for (int v_i = 0; v_i < rapidMesh.indices.size(); ++v_i)
  {
    mesh.faces[v_i] = rapidMesh.indices[v_i].position_index;
    mesh.texcoord_idx[v_i] = rapidMesh.indices[v_i].texcoord_index;
    mesh.normal_idx[v_i] = rapidMesh.indices[v_i].normal_index;
  }
}

void parse_positions(const rapidobj::Array<float> &positions, ParsedMesh &mesh)
{
  mesh.positions_shape[0] = positions.size() / 3;
  mesh.positions.resize(positions.size());
  std::copy(positions.begin(), positions.end(), mesh.positions.begin());
}

void parse_normals(const rapidobj::Array<float> &normals, ParsedMesh &mesh)
{
  mesh.normals_shape[0] = normals.size() / 3;
  mesh.normals.resize(normals.size());
  std::copy(normals.begin(), normals.end(), mesh.normals.begin());
}

void parse_texcoords(const rapidobj::Array<float> &texcoords, ParsedMesh &mesh)
{
  mesh.texcoords_shape[0] = texcoords.size() / 2;
  mesh.texcoords.resize(texcoords.size());
  std::copy(texcoords.begin(), texcoords.end(), mesh.texcoords.begin());
}

void parse_material_ids(const rapidobj::Array<int32_t> &material_ids, ParsedMesh &mesh)
{
  mesh.material_ids_shape[0] = material_ids.size();
  mesh.material_ids.resize(material_ids.size());
  std::copy(material_ids.begin(), material_ids.end(), mesh.material_ids.begin());
}

ParsedMesh parse_rapidobj(const rapidobj::Result &result)
{
  if (result.error)
  {
    throw std::runtime_error("Error loading OBJ file: '" +
                             result.error.code.message() + "'");
  }

  if (result.shapes.size() > 1)
  {
    throw std::runtime_error("Multi-shape OBJ files not supported.");
  }

  ParsedMesh parsed{};
  parse_positions(result.attributes.positions, parsed);
  parse_normals(result.attributes.normals, parsed);
  parse_texcoords(result.attributes.texcoords, parsed);
  if (result.shapes.size() == 1)
  {
    parse_material_ids(result.shapes[0].mesh.material_ids, parsed);
    parse_faces(result.shapes[0].mesh, parsed);
    parse_lines(result.shapes[0].lines, parsed);
  }
  return parsed;
}

ParsedMesh load_obj_file(std::string path)
{
  auto result = rapidobj::ParseFile(path);
  return parse_rapidobj(result);
}

ParsedMesh load_obj_string(nb::str &contents)
{
  auto stream = std::istringstream(contents.c_str());
  auto result = rapidobj::ParseStream(stream);
  return parse_rapidobj(result);
}

NB_MODULE(io_ext, m)
{
  nb::class_<ParsedMesh>(m, "ParsedMesh")
      .def(nb::init<>())
      .def("positions", [](ParsedMesh *self)
           { return Tensor<float, 3>(self->positions.data(), 2, self->positions_shape); }, nb::rv_policy::reference_internal)
      .def("normals", [](ParsedMesh *self)
           { return Tensor<float, 3>(self->normals.data(), 2, self->normals_shape); }, nb::rv_policy::reference_internal)
      .def("texcoords", [](ParsedMesh *self)
           { return Tensor<float, 2>(self->texcoords.data(), 2, self->texcoords_shape); }, nb::rv_policy::reference_internal)
      .def("material_ids", [](ParsedMesh *self)
           { return Tensor<int64_t, 1>(self->material_ids.data(), 2, self->material_ids_shape); }, nb::rv_policy::reference_internal)
      .def("faces", [](ParsedMesh *self)
           { return Tensor<int64_t, -1>(self->faces.data(), 2, self->faces_shape); }, nb::rv_policy::reference_internal)
      .def("texcoord_idx", [](ParsedMesh *self)
           { return Tensor<int64_t, -1>(self->texcoord_idx.data(), 2, self->faces_shape); }, nb::rv_policy::reference_internal)
      .def("normal_idx", [](ParsedMesh *self)
           { return Tensor<int64_t, -1>(self->normal_idx.data(), 2, self->faces_shape); }, nb::rv_policy::reference_internal)
      .def("lines", [](ParsedMesh *self)
           { return Tensor<int64_t, 2>(self->lines.data(), 2, self->lines_shape); }, nb::rv_policy::reference_internal)
      .def("__repr__", &ParsedMesh::to_string);

  m.def("load_obj_file", &load_obj_file);
  m.def("load_obj_string", &load_obj_string);
}
