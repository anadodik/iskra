# Directional

This module allows us to work with discrete vector and frame fields.
The name is a reference to Amir Vaxman's award-winning library which partially
served as inspiration for this module.

We support two types of directional fields:
* n-RoSy ($n$ rotationally symmetric) fields, sometimes also known as power fields. These represent fields as the roots of a complex number.
* Frame fields consisting of 2 directions per field element. These are represented as the coefficients of a polyvector field.

## Frame Fields

Frame fields the represented as the roots of the 4th order complex polynomial.
For each face, we store two coefficients, $c_2$ and $c_0$, which go into the polynomial:

$$
p(z) = z^4 + c_2 z^2 + c_0 = 0.
$$

This polynomial will have four roots consisting of two symmetric pairs of vectors.
We store such a polynomial as a tensor where the last dimension contains $(c_2, c_0)$ in that order.

Oftentimes, it is useful to prescribe one of the roots of this polynomial.
We do so via the method of Meekes and Vaxman, 2021 (Appendix A).
Concretely, given a symmetric 2-direction field $\bar u = u^2$, where $u$ is one of the two symmetric vectors along said direction, we can pull the root term $(z^2 - u^2)$ out of the polynomial $p$. Define $\bar z = z^2$, we have:

$$
p(\bar z) &= (\bar z^2 + c_2 \bar z + c_0), \\
p(\bar z) &= (\bar z - \bar u) (\bar z - \bar v), \\
p(z) &= (z^2 - u^2) (z^2 - v^2).
$$

We can express the coefficients $c_i$ in terms of $u$ and $v$ as:

$$
c_2 &= -v^2 - u^2 \\
c_0 &= u^2 v^2.
$$

So, having specified a $u$, we can map back and forth from $v^2$ and the full coefficients via:

$$
\begin{bmatrix}
c_2 \\
c_0
\end{bmatrix}
=
\begin{bmatrix}
- 1 & u^2 \\
u^2 & 0
\end{bmatrix}
\begin{bmatrix}
v^2 \\
- 1
\end{bmatrix}.
$$




```{eval-rst}
.. autosummary::
   :toctree: generated
   :template: module.rst
   iskra.directional

.. include:: generated/iskra.directional.rst
```


```{toctree}
:maxdepth: 1
:hidden:

generated/iskra.directional
```