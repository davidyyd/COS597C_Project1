# r13b + contextual value embeddings (bf16)
Parent: r13b-r12c-conv01-novsmear (adds r13d-context-ve). Type: exploit.

Contextual VE was the largest single gain (-0.009 alone) but slow in fp32 (r13d 294 s). Here computed in
bf16 (the value path is bf16 under autocast anyway), on the fastest strong model (r13b, 258 s).
Confirm: < 0.9413 (r12c) within the cap.
