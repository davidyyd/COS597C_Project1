# Doc-position attention temperature + MLP reads attention (new cheap lineage)
Parent: r19d-docpos-temp
Type: exploit

Combine the two strongest single changes on the baseline: doc-position attention temperature (r19d, -0.0043)
and the MLP's undiluted read of its block's attention output (r10c, -0.0052). Starts a cheaper lineage
(~66 ms) that can take more mechanisms than the r14c family. Confirm: < 0.9520.
