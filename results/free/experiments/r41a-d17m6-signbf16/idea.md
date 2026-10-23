# r40a with the sign patterns built in bf16 on the small (31/29-wide) tensors before gathering

Parent: r40a-d17-mlp6x-seed42 (best, 0.76736, 3470s total: tight). Efficiency: identical math; the int64 shift/XOR now happens on [B,T,31]/[B,T,29] and only bf16 +-1 values are gathered to [B,T,d]. Aim: some time margin.
