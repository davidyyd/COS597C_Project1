# Document masking: attention only within the same document (flex_attention, all layers)

Parent: r10d-polar-express. Explore (attention/data). Rows pack several documents (BOS-separated); tokens currently attend across document boundaries to unrelated text. Mask attention to the same document (doc id = cumsum of BOS), in both the sliding-window and full layers (all via flex_attention, BlockMask built per batch outside torch.compile). Applies at eval too (the model's own forward).
