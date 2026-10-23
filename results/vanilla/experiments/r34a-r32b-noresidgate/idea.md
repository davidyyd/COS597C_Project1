# r32b without the residual gate (keep the x0 gate), to fit the cap
Parent: r32b-r31c-novsmear. Type: exploit.

r32b (query smear + lag-2 key smear + readout) is 2.4 s over the cap at 0.934386. The per-token
residual gate is the cheaper half of the depth gates to drop; this tests its value in the full stack.
Confirm: < 0.934967 (r32a) within the cap.
