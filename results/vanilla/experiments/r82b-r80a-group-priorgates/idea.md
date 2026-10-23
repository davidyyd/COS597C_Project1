# Component-grouped context gates on the n-gram logit priors
Parent: r80a-r79a-mlpgate-late. Type: exploit.
Context-gated priors (one gate per prior order) gave -0.0008. Here 12 gates (one per 24 prior components:
8 for the bigram rank 192, 4 for the skip-bigram rank 96), so the context can select which prior
directions to trust, a low-rank context x prior interaction. Confirm: < 0.923610.
