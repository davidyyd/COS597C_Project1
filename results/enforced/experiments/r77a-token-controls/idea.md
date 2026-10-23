# Token-identity bias on every bounded per-token control
Parent: r76c-token-headgate
Type: exploit

Hypothesis: a per-token, per-(layer, head) bias on the head gate helped (r76c, -0.0003). The same argument holds for
all of the model's bounded per-token scalar controls: query temperature (how sharp to attend), x0 re-grounding
gate (how much identity to re-inject) and output temperature (how confident to be) plausibly depend on the token
identity (e.g. after "the" vs after "."). One sparse table supplies a bias for each: 73 scalars per token (598k).
Change: every control c = 1 + tanh(f(x) + B[token, c]) (or 2σ for head gates), B init 0.
Confirm: below r76c (0.93074) by > 0.0003.
