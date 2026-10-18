### The W8A16 split premise (#1313): is the attention alone inside P125's gate?

The first W8A16 premise box read UNDERPOWERED. Rounding the attention projections and the lm_head to int8 left argmax
agreement at about 0.958, with the NLL shift at 85-92 % of P125's bound. The split box separates the two parts:
attention-only int8 (the candidate) and head-only int8 (reported), at the windows the first box's spread asks for
(t1 196, k16 123). Wikitext is read every 1024 tokens with P125's own loader. The rule is fixed now: GO iff attention-only
keeps argmax >= 0.95 and the one-sided 95 % upper bound of its NLL shift within the bound at both gates, else the lever
parks. A premise, never a reading.
