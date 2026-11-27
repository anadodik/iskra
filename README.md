<p align="center">
  <a href="https://iskra-graphics.org"><img src="https://raw.githubusercontent.com/anadodik/iskra/main/docs/logo.svg" alt="iskra" width="800"></a>
</p>
<div align="center">

# `iskra` ✨ Modern Geometry Processing

</div>
<p align="center">
<a href="https://iskra-graphics.org">
    <img src="https://img.shields.io/badge/docs-iskra--graphics.org-2088c3" alt="Documentation">
</a>
<a href="https://pypi.org/project/iskra-graphics/">
    <img src="https://img.shields.io/pypi/v/iskra-graphics" alt="Package version">
</a>
<a href="https://pypi.org/project/iskra-graphics/">
    <img src="https://img.shields.io/pypi/pyversions/iskra-graphics" alt="Supported Python versions">
</a>
<a href="https://github.com/anadodik/iskra/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</a>
</p>

---

**Documentation**: [https://iskra-graphics.org](https://iskra-graphics.org)

**Source Code**: [https://github.com/anadodik/iskra](https://github.com/anadodik/iskra)

---

Lightweight geometry processing library that is a one-stop shop for all your geometric needs. Iskra is:
* modern and Python-first,
* simple by default, powerful when needed,
* fully differentiable and compatible with machine learning,
* CPU and GPU enabled,
* actievely maintained.

Support the project by starring it: 
[![GitHub stars](https://img.shields.io/github/stars/anadodik/iskra?style=social)](https://github.com/anadodik/iskra/stargazers)

> [!CAUTION]
> This is a pre-release. We are actively working on a user guide, documentation.

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

You will need PyTorch installed for `iskra` ✨ to work: [see PyTorch installation instructions here](https://pytorch.org/get-started/locally/).
In addition, if you intend on using sparse linear solvers on the GPU, you should install cuDSS via [nvmath](https://docs.nvidia.com/cuda/nvmath-python/latest/installation.html).
If you intend on using them on the CPU, you might wish to install suitesparse and [scikit-sparse](https://scikit-sparse.readthedocs.io/en/latest/overview.html) (currently only v0.4 of scikit-sparse is installed).
```bash
pip install torch --index-url ... # your preferred PyTorch distribution

# For cudss, for cuda version XX, do:
pip install nvmath[cuXX-dx]

# For cholmod, first install suitesparse using Conda or your OS distribution manager, then do:
pip install scikit-sparse
```
The code has been tested with `torch==2.12`, but will likely work with other versions too.

> [!NOTE]
> We do not include PyTorch as a dependency because the package you will install will depend on your exact setup, e.g., whether you have a GPU or not, your GPU driver version, and so on.

Finally, install `iskra` ✨ to your active environment using:
```bash
pip install iskra-graphics
```

Alternatively, if you prefer the most recent dev version:
```bash
pip install -e git+https://github.com/anadodik/iskra/
```

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
2. **How much of `iskra` ✨ is LLM generated?** Iskra started back in 2022 at the start of my PhD because I enjoy writing and learning geometry algorithms. It is therefore almost entirely good old fashioned free range human generated slop, except for a select few parts. These are clearly marked in the codebase. [Doing it this way had some benefits beyond being fun: deeply bonding with PyTorch geometry code lead me to the tensor-based scatter-gather abstraction!]