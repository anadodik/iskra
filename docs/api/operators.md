# Differential Operators

`iskra` ✨ offers two different formalisms for computing discrete differential quantities on meshes, the finite element method (FEM) and discrete exterior calculus (DEC).
While used for similar purposes, the two mechanisms offer different trade-offs, and one might be better than the other depending on the downstream application.

## FEM Basis

Today, `iskra` ✨ only supports the piecewise linear (PL) basis, also known as the "hat" finite element basis.
The PL finite element basis is a set of functions $\phi_i(x)$ defined over the mesh, one per mesh vertex indexed by $i$, which allows us to interpolate values defined on vertices onto the entire mesh. 
Given function values $f_i \in \mathbb{R}$ defined on the mesh vertices, we can use the basis to define the function $f$ at a point $x$ on the mesh as:

$$
f(x) = \sum_{i = 1}^{|V|} f_i \phi_i(x).
$$

Since the finite element basis is local, it is sufficient to consider only the face $\mathcal F$ in which $x$ finds itself:

$$
f(x) = \sum_{i \in \mathcal F} f_i \phi_i(x),
$$
where $\phi_i(x)$ are simply the barycentric coordinates of face $\mathcal F$ at $x$.

## FEM Gradients

Finite element gradients are defined on mesh faces, represented as extrinsic vectors in `iskra` ✨. 
While representing gradients as intrinsic vectors may be more general, not having to lug around a local basis can often simplify computation.
See the `iskra.directional` module for intrinsic vectors (and the more general directionals) as an alternative.


$$
\nabla f(x) = \sum_{i \in \mathcal F} f_i \nabla \phi_i(x).
$$
The piecewise-linear hat basis gradient $\nabla \phi_i(x)$ belonging to vertex $i$ in face $\mathcal F$ is defined as:

$$
\nabla \phi_i = - \frac{1}{h_i} \mathbf n_i,
$$
where $\mathbf n_i$ is the normal to the subface opposite vertex $i$ (computed in the hyperplane spanned by the face), and $h_i$ is the height of the face between subface $i$ and vertex $i$.

Using the following expression for the height of a simplex, $h_i = \frac{\text{dim}(\mathcal F) V(\mathcal F)}{V(\mathcal S_i)}$, where $\mathcal S_i$ is the subface opposite vertex $i$ and $V$ is the (hyper)volume of a face, we obtain the following form:

$$
\nabla \phi_i = - \frac{V(\mathcal S_i)}{\text{dim}(\mathcal F) V(\mathcal F)} \mathbf n_i.
$$


> [!NOTE]
> Unlike `gptoolbox` and `libigl`, `iskra` ✨ uses a face-major order for the gradient vectors.
> This allows us to compute gradients of a 2D function defined on vertices `uv` in 3D as `grad_uv = (g @ uv).reshape(n_verts, 3, 2)`, thus obtaining $3 \times 2$ stacked gradient matrix per vertex, and taking `grad_uv.mT` gives us the per-vertex Jacobian.
>
> Another difference to `gptoolbox` is that our `grad_edges` is an extrinsic edge gradient instead of an intrinsic difference matrix, thus making it consistent with the other gradient operators.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :template: module.rst
   iskra.fem
   iskra.dec

.. include:: generated/iskra.fem.rst
.. include:: generated/iskra.dec.rst
```

```{toctree}
:maxdepth: 1
:hidden:

generated/iskra.dec
generated/iskra.fem
```