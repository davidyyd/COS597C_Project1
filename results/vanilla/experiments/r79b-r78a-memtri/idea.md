# r78a with a hashed trigram lookup added to the mid-network memory
Parent: r78a-r77b-gprior-nomlpgate. Type: exploit.
r77d: adding a hashed trigram lookup into the same memory table helped the control by 3e-4; parameter-free,
~+0.5 s. In the stack the bigram is already covered by the logit priors, so the trigram is the part of the
memory with new information. Confirm: < 0.925327.
