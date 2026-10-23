# MLP also reads the previous MLP output
Parent: r12b-best-attn-reads-mlp
Type: exploit

Extend the undiluted-read chain: MLP input = norm(x) + beta norm(a_l) + delta norm(m_{l-1}) (delta init 0),
so consecutive MLPs can compose directly (deeper effective MLP) without the residual dilution.
Confirm: < r12b (0.94688) by > 0.0005.
