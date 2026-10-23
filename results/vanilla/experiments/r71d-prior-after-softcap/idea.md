# Bigram prior added after the logit soft-cap
Parent: r63d-bigram-logit-prior. Type: explore.
The prior is currently squashed together with the network's logits by the tanh soft-cap (15); adding it
after the cap lets confident bigram statistics act without saturation. (Separate (B,T,V) add: slower,
tests the effect first.) Confirm: < 0.954901 (r63d).
