# `q235-pro6000-5`: refused at the host-rate check, before the fetch; not a strike

One RTX PRO 6000 Blackwell Workstation Edition (24 CPUs, PCIe gen 5), e4b
`8ff2f5d8`. Cost $0.104. The box installed, then its host-rate check refused it: pinned host-to-device read 49.02 GB/s and
the disk 11.91 GB/s, but the host copy read 17.69 GB/s against Amendment 12's floor of 58.5. Nothing was fetched and no
step ran. A refusal is not a strike. Amendment 13 then derived the host-copy floor per host-bound part.

The launcher's receipt and ledger row are in the private receipt store. `SHA256SUMS` covers every file here.
