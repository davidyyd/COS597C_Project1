# r10c with the per-layer extras packed into a few tensors (same model, fewer all-reduces)
Parent: r10c-r09c-vsmear. Type: exploit.

Profiling (r11c/r11d) showed each parameter tensor costs ~0.15 ms/step in the per-parameter gradient
all-reduce (~82 small all-reduces per step). r10c (0.9434) added 4 small tensors per attention layer and
ran 269 s. Here the per-layer extras live in two GPT-level tensors (as the baseline does for its
lambdas): attn_scalars = [log_temp | key_smear | value_smear] per layer (1-D, AdamW, no decay as before)
and gate_bank.attn.out_gate (n_layer x 4 x 32, Muon per slice = the same update as separate 2-D gates).
Same math as r10c. Confirm: ≈ 0.9434 at < 261.7 s.
