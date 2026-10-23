# r49a + document masking in every attention layer (score_mod on all flex layers)
Parent: r49a-lean-tokmeta. Type: exploit.
Cross-document attention also happens inside the 1024-token windows; masking it everywhere may help more
(r05b: -0.0022 on the baseline with full doc masking). Cost: the doc score_mod on every flex layer.
Confirm: < r53a within the cap.
