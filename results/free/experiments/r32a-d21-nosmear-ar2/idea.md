# r30a without smear + leaner bf16 grad all_reduce (no fp32 temporary)

Parent: r30a-d21-r28b-seed42 (best, 0.77256); retry of r31a (OOM at step 0 in the flat all_reduce: torch.cat made a 6.4GB fp32 temporary). Now grads are cast directly into a preallocated bf16 buffer. Smear removed (neutral at d16).
