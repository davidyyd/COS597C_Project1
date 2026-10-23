# Feature-grouped gate on the mid-network hashed memory
Parent: r80a-r79a-mlpgate-late. Type: exploit.
The memory read is gated by one scalar per token; gate 8 groups of the 128 memory features separately
(before the projection), so the state picks which memory features to use. Confirm: < 0.923610.
