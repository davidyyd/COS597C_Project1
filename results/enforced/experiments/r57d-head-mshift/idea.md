# LM head reads the last MLP output of the previous position
Parent: r54c-r51d-k-shift-only
Type: explore

Time-shifted reads helped inside the network. At the output, the head reads norm(x_L) + beta norm(m_L)[t-1]
(the last MLP's conclusion about the previous token). Same-position head reads hurt (r12a); this tests whether
the previous position's conclusion is useful. Confirm: < r54c (0.93612).
