# d16w1792 canon + Muon fix (static compile, one bf16 all_reduce of updates)

Parent: r08a-d16w1792-canon. Exploit (efficiency). r08b timing: Muon 204 ms/step = NS on square matrices 55ms (dynamic-shape recompiled kernel) + ~140ms in per-chunk all_gathers. Fix: muon_update compiled with dynamic=False (cache limit 64), and updates exchanged via one all_reduce of a flat bf16 buffer (each rank fills its share); all ranks apply identical updates. Expect ~150ms/step saved; loss ~same as 0.79463 (bf16 rounding of the update).
