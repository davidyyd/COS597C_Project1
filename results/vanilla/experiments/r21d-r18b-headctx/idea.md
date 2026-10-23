# r18b + contextual x0 skip into the output head
Parent: r18b-r17a-bf16. Type: exploit.

r03c showed a small gain (-0.0018) from re-adding the plain x0 before the head. With a contextual x0 the
skip carries the local n-gram directly to the head (n-gram statistics for next-token prediction).
Change: logits = W norm(norm(x_L) + λ x0_ctx), λ init 0. One extra pass at the end.
Confirm: < 0.9369 within the cap.
