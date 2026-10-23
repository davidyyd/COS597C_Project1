# d16 (r35d): input-dependent gates on the U-net skip connections

Parent: r35d-d16proxy-xsa-trigramout (0.77668). Explore (MUDD-lite, modded-nanogpt #81/#85 use dynamic dense connections): each U-net skip is scaled by 2*sigmoid(w_j . x[:32]) (zero init = gate 1) in addition to its learned scalar.
