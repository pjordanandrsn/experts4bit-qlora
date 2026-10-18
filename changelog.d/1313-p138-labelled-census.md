### P138: a labelled census reading is not a failure (#1313)

P138's first box-started attempt (`vsplit-5090-2`) stopped at its canary, although the canary was clean. SC1b's
census reader marks a reading `labelled` whenever it attaches any label. The canary always carries
`NO_UNPROFILED_TIMING`, because no unprofiled time exists before timing pass A. The runner's canary check and the
reducer's census check both required status `ok`. The registered rule fails a reading only when it is void or carries
one of the four blocking labels. Both checks now follow it, so the reporting labels (`NO_UNPROFILED_TIMING`,
`PROFILER_INFLATED`, `NODE_TRACE_INFLATED`) read through. The box's own canary records are the tests' fixtures: the
runner's canary block passes them, and the old rule fails them as it did on the box. The reducer's self-test census
now marks a labelled reading `labelled`, as the real reader does.
