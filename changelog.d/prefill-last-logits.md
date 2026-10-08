### Opt-in final-position prefill logits

`E4B_PAGED_LAST_LOGITS=1` asks a model with an explicit supported keyword to
project only the final prompt position through its LM head. All prompt tokens
still populate K/V. Eager prefill, graph capture and its startup check share the
mode; `/health` reports the keyword and forward count. Unsupported forwards
refuse. The default remains off pending a served-prefill quality and speed read.
