# d16 (r35d): cautious weight decay for Muon

Parent: r35d-d16proxy-xsa-trigramout. Explore (optimizer): cautious WD (used in modded-nanogpt's Muon): decay only coordinates where the update and the parameter have the same sign (the update pushes toward zero anyway), leaving the others undecayed.
