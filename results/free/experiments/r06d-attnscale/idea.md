# Attention logit scale 0.12 (instead of 1/sqrt(128)=0.088) after QK-norm

Parent: r05d-smear. Explore (architecture). With gain-free QK-norm the attention logits are capped at ~sqrt(d)*scale; modded-nanogpt found a larger scale (0.12) better, letting heads attend more sharply.
