# r29a + token shift on the MLP input
Parent: r29a-r27b-headchan-gain (adds r29c). Type: exploit.

MLP input token shift (per-channel lag-1 smear) gave -0.0024 on the control at ~4 s; the stack has
local context in embeddings, keys and values but the MLPs still see one position.
Confirm: < 0.935361 within the cap.
