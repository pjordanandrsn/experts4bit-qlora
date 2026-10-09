### FAM Amendment 9 (#1362): Mixtral's and Gemma-4's runs launch past #1482

`fam-mixtral-3` VOIDed on its first forward: #1477's MoE residual fold raised a TypeError in the hybrid tier that the
default server installs, and #1482 fixed it. The remaining FAM runs now launch from this amendment's merge. The rule,
tables and pins are unchanged. On a tiny Mixtral's served build, the glue-kernel calls per decode step are identical
before and after the change and match the registered table. No second Mixtral proof is planned.
