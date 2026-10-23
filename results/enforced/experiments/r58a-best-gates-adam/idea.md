# Small gate matrices labelled as gates ("other", AdamW) instead of attention projections (Muon)
Parent: r56b-g37-kshift-mread (0.93499)
Type: exploit

The value-path gates (ve_gate, proj_gate: [4, 512]) and the head self-gate vectors (out_gate: [4, 128]) are gates,
not projections; CLAUDE.md lists gates under "other". Labelled "attention" they are trained by Muon, which moves
every entry of a small matrix by ~0.2 x LR per step (fast, orthogonalized gate changes). Label them "other"
(AdamW). Confirm: < r56b.
