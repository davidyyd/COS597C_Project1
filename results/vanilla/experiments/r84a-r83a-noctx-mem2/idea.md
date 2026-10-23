# r83a - ctx-x0 conv + second memory read (skip-bigram hash) at layer 6
Parent: r83a-r82c-group-priorgates. Type: exploit.
Dropping the ctx-x0 conv costs only 2.7e-4 and frees 1.5 s (r83b). Spend it on a second read of the hashed
memory at layer 6, keyed by the skip-bigram (x_{t-2}, x_t) (shared table and projection; own grouped gate;
parameter cap leaves no room for a new table). Confirm: < 0.923469 within the cap.
