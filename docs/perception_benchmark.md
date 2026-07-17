# Perception experiment

Benchmark command:

```bash
python scripts/benchmark_perception.py --steps 100 --nca-steps 32
```

Environment: JAX 0.6.2 on `cuda:0`, 56×56 grid, batch size 8, 100 training
steps, and 32 NCA steps per training/evaluation pass. The first-call timing
includes XLA compilation; steady-state timing excludes it.

| Method | Compile + first step | Steady-state step | Final MSE |
| --- | ---: | ---: | ---: |
| `sobel` | 14.94 s | 15.08 ms | 0.1371600 |
| `sobel_fused` | 13.91 s | 14.11 ms | 0.1371776 |
| `sobel_second` | 13.76 s | 8.73 ms | 0.0873381 |
| `learned` | 15.21 s | 13.17 ms | 0.1830653 |

For a state channel \(u(x,y)\), the first- and second-order derivatives are

$$
u_x = \frac{\partial u}{\partial x}, \qquad
u_y = \frac{\partial u}{\partial y}, \qquad
u_{xx} = \frac{\partial^2 u}{\partial x^2}, \qquad
u_{yy} = \frac{\partial^2 u}{\partial y^2}.
$$

The second-derivative kernels used by `sobel_second` are

$$
K_{xx} =
\begin{bmatrix}
0 & 0 & 0 \\
1 & -2 & 1 \\
0 & 0 & 0
\end{bmatrix},
\qquad
K_{yy} =
\begin{bmatrix}
0 & 1 & 0 \\
0 & -2 & 0 \\
0 & 1 & 0
\end{bmatrix}.
$$

The resulting perception vector is

$$
P(u) = \left[u,\; u_x,\; u_y,\; u_{xx},\; u_{yy}\right].
$$

For \(C=16\) state channels, this produces \(5C=80\) perception channels.
`sobel_fused` computes the same X and Y Sobel features in one convolution.
`sobel_second` gives the update network curvature information as well as
first-order gradients. In this run it was both faster and more accurate than
the first-order variants. The learned 3×3 perception block is trainable and
valid, but needs a longer or better matched training schedule before it is
competitive on accuracy.

## Multi-scale experiment

The new `sobel_multiscale` mode combines centered 3×3 and 5×5 Sobel-like
filters in one convolution:

$$
P_{\mathrm{multi}}(u) =
\left[u,
u_x^{(3)}, u_y^{(3)},
u_x^{(5)}, u_y^{(5)}\right].
$$

The 5×5 derivative filters are constructed separably from

$$
s = [1,4,6,4,1], \qquad d = [-1,-2,0,2,1],
$$

with

$$
K_x^{(5)} = s^\mathsf{T}d,
\qquad
K_y^{(5)} = \left(K_x^{(5)}\right)^\mathsf{T}.
$$
The 3×3 filters are zero-padded into the center of the 5×5 kernel before the
fused convolution.

A full CUDA benchmark at 56×56, batch size 8, 100 training steps, and 32 NCA
steps per pass produced:

| Method | Step time | Final MSE |
| --- | ---: | ---: |
| `sobel_fused` | 9.87 ms | 0.1372 |
| `sobel_second` | 8.25 ms | 0.0838 |
| `sobel_multiscale` | 9.52 ms | 0.1171 |
| `learned` | 9.78 ms | 0.1831 |

The multi-scale mode improves accuracy over first-order Sobel, but remains
slower and less accurate than the second-derivative mode in this run.

## Global non-local connection experiment

The architecture-level non-local branch is enabled with
`nonlocal_connections: true`. It computes a global summary of the local
perception and broadcasts a learned projection back to every cell:

$$
g = \frac{1}{HW}\sum_{i=1}^{H}\sum_{j=1}^{W} p_{ij},
\qquad
z_{ij} = \left[p_{ij},\; \phi(g)\right],
$$

where \(p_{ij}\) is the local perception at cell \((i,j)\), and
\(\phi: \mathbb{R}^{C}\rightarrow\mathbb{R}^{32}\) is a learned dense
projection. Thus every cell receives information derived from the entire grid.
This is the architecture-level long-range experiment.

20-step CPU smoke benchmark at 32×32:

| Method | Step time | Final MSE |
| --- | ---: | ---: |
| `sobel_second` | 151.66 ms | 0.1799 |
| `global_context` | 157.58 ms | 0.1914 |

The global branch is functional, but this short run does not yet show an
accuracy benefit. A full CUDA training run is needed for a stronger comparison.
