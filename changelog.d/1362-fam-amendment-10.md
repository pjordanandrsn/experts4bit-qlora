### FAM Amendment 10 (#1362): Gemma-4's per-step glue norms by shape

`fam-gemma4-prove-1` VOIDed on engagement: at T == 12 Gemma-4 called 216 glue norms per decode step against the
registered 271. The glue fold's 64-row bound sends the per-head q and k norms to their own torch forward once
sequences × heads passes 64. That is a re-route, not a skip: every norm still runs, and the outputs stay at rounding
level. The reducer now counts Gemma-4's ON_glue and ON_auto per shape: 216 at T == 12 and 271 at T == 1.
