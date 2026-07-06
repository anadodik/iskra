![](docs/logo.svg)

# `iskra` ✨ Modern Geometry Processing


Lightweight geometry processing library that is a one-stop shop for all your geometric needs. Iskra is:
* modern and Python-first,
* simple by default, powerful when needed,
* fully differentiable and compatible with machine learning,
* CPU and GPU enabled,
* actievely maintained.

Support the project by starring it: 
[![GitHub stars](https://img.shields.io/github/stars/anadodik/iskra?style=social)](https://github.com/anadodik/iskra/stargazers)

> [!CAUTION]
> This is a pre-release. We are actively working on a user guide, documentation, and setting up on PyPI.

## Example

Computing vertex normals in iskra is as simple as:
```python
import torch

from iskra.geometry import triangle_normals
from iskra.mesh import Mesh
from iskra.topology import face_index, reduce_on_subface

mesh, _ = Mesh.from_path(
    "oded://objects/koala/koala_low_resolution.obj",
    device="cpu"
)
verts = mesh.vertices  # [V, 3]
faces = mesh.faces  # [F, 3]

tris = face_index(verts, faces)  # [F, 3, 3]

tri_normals = triangle_normals(tris)  # [F, 3]

vert_normals = reduce_on_subface(tri_normals, faces, verts.shape[0], "sum")  # [V, 3]
vert_normals = torch.nn.functional.normalize(vert_normals, dim=-1)  # [V, 3]
```

## Obtaining `iskra` ✨

Install `iskra` ✨ to your active environment using:
```bash
pip install -e git+https://github.com/anadodik/iskra/
```

You will need PyTorch installed for `iskra` ✨ to work: [see PyTorch installation instructions here](https://pytorch.org/get-started/locally/).
The code has been tested with `torch==2.12`.
It will likely work with other versions too.

> [!NOTE]
> We do not include PyTorch as a dependency because the package you will install will depend on your exact setup, e.g., whether you have a GPU or not, your GPU driver version, and so on.

## Development
Lastly, if you plan on contributing, you will need the development dependencies and to compile the C++ extensions in editable mode.
This can be done by running the following:
```
conda env create -f environment.yaml
conda env update -f environment-dev.yaml
conda activate iskra
pip install --no-build-isolation -Ceditable.rebuild=true -ve .
```

## FAQ
1. **Why the name?** Iskra means “spark” in Serbo-Croatian: a spark enables using (a) torch. We also expect our system to be the spark that ignites exciting research in geometry. Most importantly, it sounds cool.