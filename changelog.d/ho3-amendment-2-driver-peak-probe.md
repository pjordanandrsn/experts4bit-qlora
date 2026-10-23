### HO3 Amendment 2: a driver-peak probe before the install

- A box now refuses before its install (rc 19, a host refusal counting toward nothing) when Loggetta 0.6.2's own driver
  sampler, staged byte for byte and checked against the installed module, cannot sample this process's GPU memory. Such
  a host would otherwise read VOID after its whole run.
- `ho3_reduce.py --classify` names the runner's pre-install exits. The driver stops a card after three consecutive rc 19.
