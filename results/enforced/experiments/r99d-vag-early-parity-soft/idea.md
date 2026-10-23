# Learned soft document isolation in early layers (parity q/k features instead of the hard mask)
Parent: r88c-vag-doc-parity
Type: explore
Can a learned bias replace the hard early mask? Early local layers get q += s_t p_q, k += s_j p_k (s = +-1 per
document, zero-init), no hard mask. Compare with r83a (hard mask, 0.94006) and r58b-like unmasked (~0.9419+).
