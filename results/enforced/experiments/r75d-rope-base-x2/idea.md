# Control for the RoPE rule: base = 20 x window (twice r74b's) in every head
Parent: r74b-rope-scale-full
Type: exploit (mechanism test)

Question: r73d/r74b set each head's RoPE base proportional to its receptive field (base ≈ 10·w) and improved
slightly. Is the gain from MATCHING base to window (proportionality) or from the overall level? Same rule with
the constant doubled (bases 2500..20000 in sliding heads, 40000 in full layers).
Confirm "level matters": clearly better than r74b. Neutral: proportionality is what matters, constant not critical.
