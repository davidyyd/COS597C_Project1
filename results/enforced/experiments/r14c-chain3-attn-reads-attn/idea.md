# Attention also reads the previous attention output (no prev-token keys)
Parent: r13c-mlp-reads-prevmlp
Type: exploit

Complete the chain: attention input = norm(x) + gamma norm(m_{l-1}) + eps norm(a_{l-1}), so successive
attention layers can compose (e.g. induction: layer l's query reads what layer l-1 retrieved) without
residual dilution. Prev-token keys removed to stay under the step-time cap (see r14b for that ablation).
Confirm: < r14b.
