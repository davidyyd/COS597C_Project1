# Channel-grouped stale depth gates
Parent: r80a-r79a-mlpgate-late. Type: exploit.
The stale depth gates (per-token scalar gates on x0 re-injection and the residual) are worth 1.6e-3 (r81c).
Make them per channel group (4 groups of 128): the token decides how much of each part of x0/residual to
keep. Same single small matmul (32 -> 8 instead of 32 -> 2). Confirm: < 0.923610 within the cap.
