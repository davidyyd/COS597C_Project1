# r18b + word-so-far embedding in the contextual x0
Parent: r18b-r17a-bf16 (adds r24c's word-so-far embedding). Type: exploit.

Adds the variable-span word-prefix mean to the fixed-lag contextual x0 of r18b.
Confirm: < 0.9369 (cap may be exceeded: one cumsum/gather pass at the input).
