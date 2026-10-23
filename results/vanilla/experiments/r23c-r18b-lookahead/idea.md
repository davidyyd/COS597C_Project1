# r18b + lookahead value embeddings in layer 7
Parent: r18b-r17a-bf16 (adds r23a's mechanism). Type: exploit.

Tests the lookahead values in the best stack (which already has smeared keys, a related key-side trick).
Expect it to exceed the time cap (an extra full-attention pass); this measures its value in the stack.
