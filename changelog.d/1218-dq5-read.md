### DQ5 read (#1083): PROTO_PASS on PCIe gen 4 x16 — streamed QLoRA at 1.0050× resident, bitwise, 15.08 GB freed (bench only; #1218)

- **The run.** `dq5-5090-1` ($0.401) ran DQ3's lane and rule on a gen 4 x16 RTX 5090. The link was confirmed x16
  under load, at about 21–22 GB/s.
- **The gates.** T(S)/T(R) = 1.0050, inside the host's step-to-step noise. 0 blocking prefetches, 15.08 GB saved,
  parity bitwise.
- **Thinner margin than gen 5.** 20–59 of 62 forward copies per step were still in flight when their layer started
  (gen 5: 9–13). The synchronous path pays 1.51× (gen 5: 1.18×).
- **Evidence.** The receipts re-derive the verdict byte for byte. `bench/dq5/RESULTS-dq5.md` carries the host context
  and an independent review.
