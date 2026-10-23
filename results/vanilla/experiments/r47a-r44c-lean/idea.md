# r44c with 2-lag ctx-VE and bf16 ctx-x0 conv (lean base)
Parent: r44c-r42b-unet (combines r46a and r46b). Type: exploit (cost reduction).
Each saved 5-10 s at no measurable loss; together they free ~15 s for other components.
Confirm: ≈ 0.932 at ~244 s.
