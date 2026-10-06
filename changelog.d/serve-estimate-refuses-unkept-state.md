### Serve estimate: refuses the hybrid models the paged runner refuses

- `estimate_serve_footprint` planned serves for LFM2-8B-A1B and granite-4.0-h-tiny that `serve_paged` refuses when it
  builds its runner:
  - LFM2's `conv` layers are a type the runner keeps no state for;
  - granite-4.0-h's Mamba layers are labelled `linear_attention`, but the per-slot state pool drives Gated DeltaNet
    only.
- `engines.paged_runner.paged_state_refusal(model)` states both of the runner's rules (`layer_plan`'s kept layer types
  and `linear_state.install`'s drivable layers, through the new `linear_state.driven_linear_layers`) on a meta tree.
- `MoETopology.paged_state_refusal` carries that verdict, and the estimate refuses with it.
- A test builds each tiny model for real and checks that the runner's layer plan refuses exactly when the verdict says
  so.
