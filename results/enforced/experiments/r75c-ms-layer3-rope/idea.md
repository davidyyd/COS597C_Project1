# Multi-scale heads in the middle full-attention layer (with RoPE base ∝ window)
Parent: r74b-rope-scale-full
Type: explore

Hypothesis: in r74b each head's RoPE base matches its receptive field. Layer 3 (full attention) still has four
identical full-context heads; giving it the multi-scale structure too (windows 256/512/1024/2048, one head keeps
the whole context, bases 2500..20000) diversifies its scales like the sliding layers.
Change: layer 3 windows 2048 >> (3-h) per head, matched RoPE bases; layer 7 stays full.
Confirm: below r74b by > 0.0005. Refute: worse (the middle layer needs several long-range heads).
