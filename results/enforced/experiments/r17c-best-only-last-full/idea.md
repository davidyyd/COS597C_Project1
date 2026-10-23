# Only the last layer has full attention
Parent: r14c-chain3-attn-reads-attn
Type: exploit (ablation / time)

With undiluted reads chaining layers, layer 3's full attention may be replaceable by a multi-scale sliding
layer (pattern S...S + last layer full). Saves time. Confirm "fine": within +0.0007 of r14c.
