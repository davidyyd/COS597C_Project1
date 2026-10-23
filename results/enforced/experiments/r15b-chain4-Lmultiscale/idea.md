# Multi-scale heads in the full-attention layers too
Parent: r14c-chain3-attn-reads-attn
Type: exploit

Multi-scale head windows in the sliding layers helped (r02d, r09a). Apply the same per-head geometric windows
in the full layers (256, 512, 1024, 2048): every layer then has local-to-global heads, one head still global.
Also saves time (needed to re-add prev-token keys, see r15c). Confirm: <= r14c (0.94455).
