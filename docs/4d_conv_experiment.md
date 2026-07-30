# Conv4D experiment

This branch adds a four-spatial-dimensional NCA with state layout
`N,C,Z,Y,X,A`. The `A` axis is spatial; rollout steps remain the NCA time
axis. `UpdateModel4D` uses a local `(3, 3, 3, 3)` Conv4D perception kernel and
pointwise update layers.

The default 2D reconstruction is `project_xy_over_a_4d`: it selects one `Z`
slice and alpha-composites all `A` slices into an RGBA image. This is preferred
for the first experiment because every `A` slice receives reconstruction
gradients. `render_xy_plane_4d` is available for inspecting one exact `(Z,A)`
hyperplane.

The initial smoke dimensions are deliberately small (`Z=4`, `Y=X=8`, `A=3`).
Scale only after the shape, projection, and alive-mask tests pass; memory and
compute grow approximately linearly with the new A axis.
