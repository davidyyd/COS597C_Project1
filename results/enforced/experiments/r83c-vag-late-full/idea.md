# Late layers attend over the whole row (no sliding window in the second half)
Parent: r81c-vag-dm012-doctemp
Type: explore
Hypothesis: early layers want only the current document (doc masking 0-2 helps), late layers want other documents
(masking them hurts). Then the late local layers' 1024-token window may also be too short: give layers 4-6 full
context (the first half keeps windows + document masks). Locality should decrease with depth.
Confirm: below r81c (0.94189) by > 0.0005.
Closest work: Longformer (window size grows with depth); here motivated by the document-isolation depth split.
