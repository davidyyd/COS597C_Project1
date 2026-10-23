# r40a with per-channel residual and x0 mixing weights
Parent: r40a-r39b-noptr-noreadout. Type: exploit.
The x0 re-injection and residual carry use one scalar per layer; making them per-channel vectors lets each
layer choose which embedding features to re-inject. It reuses the existing mixing op (no extra pass).
Confirm: < 0.932561 within the cap.
