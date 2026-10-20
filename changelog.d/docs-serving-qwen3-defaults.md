### Docs: the serving reference states Qwen3's three per-family decode defaults

`docs/SERVING.md` now documents three settings that resolve per family at startup:
- the attention split cap (`E4B_ATTN_SPLIT_CAP`, 16 on `qwen3_moe`, lane P143);
- the int8 output head (`E4B_INT8_HEAD_BW`, on for `qwen3_moe` at its 151,936 x 2,048 head, lane P153);
- K25's byte-permute codebook decode (`E4B_K25_LUT=prmt` on `qwen3_moe`, lane P158).

Each entry gives the measured claim id, the opt-out and the `/health` field that reports the resolution. They were
defaults on main, but the serving reference did not name them.
