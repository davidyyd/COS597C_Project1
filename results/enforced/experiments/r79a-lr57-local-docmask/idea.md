# Best + local document masking, low-rank head gate kept in layers 5 and 7
Parent: r77b-token-head-mlp
Type: exploit
Like r79b (local doc masking paid for by removing the low-rank head gate), but keeping the low-rank gate in the
two late odd layers, using the remaining time budget.
Confirm: below r77b (0.93066) and r79b.
