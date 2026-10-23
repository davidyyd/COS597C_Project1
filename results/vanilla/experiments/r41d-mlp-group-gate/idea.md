# Grouped low-rank gate on the MLP hidden units (one gate per 16 units)
Parent: r25a-bf16-resid. Type: explore.
The per-unit low-rank hidden gate helped (r39c, -0.0034) but cost 18.5 s. Gating groups of 16 hidden
units (128 gates from a 32-channel read) keeps token-dependent feature-group selection at a fraction of
the gate cost. Confirm: close to r39c's gain at much lower cost. Closest work: GLU variants, MoE-style
group routing (soft, dense).
