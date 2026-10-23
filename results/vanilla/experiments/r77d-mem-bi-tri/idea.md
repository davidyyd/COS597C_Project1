# Mid-network gated memory with hashed bigram + trigram lookups
Parent: r76d-midnet-ngram-memory. Type: explore (refine of a new direction).
r76d (gated hashed bigram memory at layer 4) gave -0.001 vs control. Add a hashed trigram lookup into the
same 16384x128 table (summed before the projection): longer n-grams are what attention handles less
cheaply. Confirm: < 0.95623. Closest: Engram multi-order n-gram memory; here a single shared table.
