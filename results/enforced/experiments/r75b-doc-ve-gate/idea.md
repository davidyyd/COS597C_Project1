# Value-embedding share depends on document position
Parent: r74b-rope-scale-full
Type: explore

Hypothesis: early in a document a token's meaning is mostly its identity; later the contextual projection
carries more. The VE gate must infer document position from the residual. Feeding f_doc to the VE gate lets the
token-identity share of each head's values follow document position directly (document position helped as query
temperature, output temperature, x0 gate).
Change (VE layers): g_ve = 2σ(W_ve x + c_h f_doc), c_h per head init 0.
Confirm: below r74b (0.93121) by > 0.0005.
