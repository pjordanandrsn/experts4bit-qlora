### P151 Amendment 1 (#1313): three spare run ids for refusals before the install

`p151-5090-1` was refused by the box's 575 W power floor (rc 17, a 520 W card with no recorded cap), and `p151-5090-2`
at the provider (`no_such_ask`), both at $0 and before any install. P151 now also accepts `p151-5090-4`, `-5` and `-6`.
They are only for refusals before the install: a host floor (rc 17 or 15), a pre-create refusal, or `no_such_ask`. A box
that starts and fails is still a strike. The runner's and the driver's run-id lists, the staged pin and the tests follow;
the rule and the box terms are unchanged.
