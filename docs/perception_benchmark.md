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
