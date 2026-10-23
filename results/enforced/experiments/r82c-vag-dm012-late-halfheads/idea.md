# Early local layers document-isolated; late local layers: half the heads isolated
Parent: r81c-vag-dm012-doctemp
Type: exploit (refines the depth split)
Question: late local layers need cross-document access (masking them all hurt, r79d), but maybe only some heads
need it. Here, in layers 4-6, heads 0-1 are document-masked and heads 2-3 keep cross-document access.
Confirm: below r81c (0.94189): isolation helps per head, cross-document access is needed only by some heads.
