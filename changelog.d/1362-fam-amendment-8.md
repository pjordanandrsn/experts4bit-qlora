### FAM Amendment 8 (#1362): Mixtral gates on × 0.90, then × 0.80

`fam-mixtral-prove-2` showed that × 0.90 barely moves Mixtral: it passed the gate on c4val1 at proof scale. Mixtral's
OFF process now also scores the pre-registered × 0.80 rung on every set. The claimed resolution is the weakest rung
that fails every gated cell, so a PASS reads as "no effect as large as a × 0.90 (or × 0.80) softmax-scale change". The
reducer reports the resolution for every family. Mixtral's reading guard is now 6.0 h, sized from the proof's measured
times.
