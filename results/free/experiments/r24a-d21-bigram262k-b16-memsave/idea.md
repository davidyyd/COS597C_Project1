# d21 bigram 262k (r23a) at device batch 16: bf16 Muon momentum + per-step Muon update buffer

Parent: r23a-d21-bigram262k-wd0-bf16ar-b8 (best, 0.77532, but 3534s total). Efficiency: get back to batch 16 (~90s faster) by freeing ~3.2GB at the backward peak: Muon momentum stored in bf16 (it is orthogonalized anyway), and the flat bf16 update buffer allocated per step instead of kept (freed before the next forward). Risk: OOM (~78GB est.).
