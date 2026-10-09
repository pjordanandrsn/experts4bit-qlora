### Gpt-oss pipelined residency follows its base routing keyword names

The gpt-oss pipelined forward now accepts `top_k_index` and `top_k_weights`,
matching its residency base. Existing positional callers and the numerical
body are unchanged. The CPU signature guard is strict for every residency
class; its former named exception is removed and obsolete caller keywords
are checked from source.
