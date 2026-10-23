# r71c with factorised (rank-256) value-embedding tables
Parent: r71c-noVE3-split192-96 (0.925878 @ 262.2) + r72d's factorised VE. Type: exploit.
Factorised VE tables were better than full ones on the control with half the parameters; smaller lookups
may also save the 0.5 s r71c is over. Confirm: < 0.926560 within the caps.
