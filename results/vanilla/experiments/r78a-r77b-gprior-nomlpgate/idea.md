# r77b + context-gated priors, paid for by dropping the MLP hidden gate
Parent: r77b-r69a-nounet-mem. Type: exploit (combine).
r77b (no U-net + gated mid bigram memory, 0.925304) and r77a (context-gated priors, 0.925797) both beat
r69a but ran 262.5 s (over the cap). The MLP hidden gate is worth ~6e-4 for ~3.2 s (r76b), so drop it and
add the prior gate: expected ~0.9250 at ~260 s. Confirm: valid and < 0.92658.
