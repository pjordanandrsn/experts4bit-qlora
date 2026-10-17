### Added

- `bench/expert-tiling/`: an analytic note, from existing records, on streaming routed experts host-to-device in sub-expert weight tiles against whole layers, routed whole experts and layer-major. Tiling moves no fewer bytes, and in training it adds 50 % unless tiles stay resident. Its time gain over whole-layer double buffering is the pipeline fill, about 1 % of a Q235 Shape B pass. The note deprioritizes it behind layer-major, routed staging in training and double buffering. Documentation and one CPU script; no runs.
