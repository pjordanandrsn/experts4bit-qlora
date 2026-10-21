### D0 Amendment 1: the fixtures' data

Loggetta's built-in Alpaca demonstration cannot supply the fixtures' 200 × 512 tokens at `90a7697d`: row 284 has an
empty output, and the demonstration's streamed path refuses it before any weight loads. The fixtures therefore train on
Alpaca's own `text` column at a pinned revision (P224 Amendment 1). D0's held-out scoring and its unseen-row proof
follow that data. Nothing else in D0 moves, and no number exists yet.
