# r35d + pointer-sentinel copy head (fp32 scores, causal query chunks)
Parent: r35d-r32a-selective-smear (adds the pointer of r34c/r34d). Type: exploit.

The pointer gave -0.0008 in the stack (r34d) but cost 12 s. Computing it over 4 causal query chunks
(chunk c scores only keys before its end) skips ~37% of the (T x T) score entries; fp32 scores are kept
(bf16 scores hurt, r35a). Confirm: < 0.934208 within the cap.
