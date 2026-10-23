# Bag-of-words document context in x0
Parent: r25a-bf16-resid. Type: explore.
Hypothesis: every block re-reads x0 (the token embedding); giving x0 a cheap, order-free summary of the
current document (topic) lets all layers condition on it without spending attention on it.
Change: m_t = mean of wte(x_s) for s from the last BOS <= t to t (cumsum difference, causal);
x = norm(wte(x_t) + P norm(m_t)), P 512x512 zero-init. Confirm: < 0.95726. Refute: attention already
provides document context (no gain). Closest: continuous cache / topic vectors (TopicRNN), linear
attention with uniform weights; difference: a doc-reset running mean folded into the re-injected x0.
