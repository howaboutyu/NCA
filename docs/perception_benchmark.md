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

`sobel_fused` computes the same X and Y Sobel features in one convolution.
`sobel_second` adds (u_{xx}) and (u_{yy}) to the perception vector using
3×3 finite-difference kernels, giving the update network curvature information
as well as first-order gradients. In this run it was both faster and more
accurate than the first-order variants. The learned 3×3 perception block is
trainable and valid, but needs a longer or better matched training schedule
before it is competitive on accuracy.
