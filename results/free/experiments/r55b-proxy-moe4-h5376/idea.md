# proxy with 4-expert MoE MLP (expert hidden 5376 = 3x, cf 1.1)

Parent: r45d. Explore: 4 top-1 experts of hidden 3x: ~dense-3x MLP FLOPs (x1.1) with 12x-equivalent MLP params. Compare against dense 6x (r54b 0.76734) and MoE 2x6x (r55a) for quality per time.
