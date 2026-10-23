# Writable x0 stream: blocks may add their output into the re-injected embedding stream
Parent: r47a-r44c-lean. Type: exploit/explore.

The re-injected x0 (contextual embedding) is constant across depth. Letting each block also write a learned
fraction of its output into the x0 stream (x0 <- x0 + c_i (block output), c init 0) turns it into a second,
slowly-updated residual stream that later layers re-read through the existing gated x0 injection
(a 2-stream hyper-connection specialised to the x0 path). Cost: one bf16 add per layer.
Confirm: < 0.932349 (r47a) within the cap. Closest work: hyper-connections (Zhu et al. 2024), DenseFormer.
