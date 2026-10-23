# r79a: halve the hashed memory table, spend the parameters on bigram-prior rank (192 -> 256)
Parent: r79a-r78a-dropve3. Type: exploit (parameter reallocation between two mechanisms, at the cap).
The memory table (16384x128) and the bigram prior both carry bigram statistics; the prior has been the
most parameter-efficient. Confirm: < 0.924342.
