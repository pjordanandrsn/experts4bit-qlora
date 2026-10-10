### Record loaded adapter paths for future dense comparisons

Add an opt-in bench helper that records adapter storage dtypes and loaded forward
and attention-binding identities, source hashes and runtime code hashes. CPU
controls detect replaced forwards. Execution and per-operation precision remain
explicitly unobserved; existing TC1 harnesses and receipts are unchanged.
