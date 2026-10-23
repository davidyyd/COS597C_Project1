# Ablation of r82c: no contextual x0 conv (x0 = norm(token embedding + metadata))
Parent: r82c-r80a-group-memgate. Type: exploit (pricing). The ctx-x0 conv was worth -0.0057 on the control
early on; with the n-gram priors, the hashed memory and the q/k smears it may now be redundant.
