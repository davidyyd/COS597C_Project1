# Best: MLP reads the attention output at t and t-1 (instead of the previous MLP output)
Parent: r52a-gate-shift-swap (0.93572)
Type: exploit (time-neutral swap)

The MLP's read of this block's attention output at the previous position helped (r52c, -0.0006 on r51d). Swap it
in for the MLP's read of the previous block's MLP output. Confirm: < r52a.
