# Correctness and memory only

These receipts were produced on an RTX A2000, a correctness-only testbed (owner policy, re-stated 2026-10-05 on
pjordanandrsn/experts4bit-qlora#835). Their loss/gradient equivalence results and their allocator peak bytes are
evidence; their timing fields (`ms`, `s`) are recorded as produced but are **not speed evidence** and are quoted nowhere.
The chunked loss's time cost on a target card is a registered TC1 lane's to read.
