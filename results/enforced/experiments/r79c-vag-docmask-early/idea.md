# Document masking only in the early local layers (0-2)
Parent: r78d-vag-local-docmask
Type: explore
Question: r78d (doc masking in all sliding layers, -0.0019) vs which depth needs it. Hypothesis: early layers build
local token features, which cross-document keys corrupt most; later local layers may use other documents.
Change: sliding layers 0-2 masked, 4-6 not. Compare with r79d (late only) and r78d (all).
