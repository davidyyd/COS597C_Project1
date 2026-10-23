# Document position everywhere it helped + RoPE bases x2
Parent: r75a-rope-doc-both
Type: exploit
Hypothesis: document position helped (slightly) as query temperature, output temperature, x0 gate input and VE
gate input; plus RoPE base 20 x window. One model with all of them tests whether the small gains add up.
Change: r75a + f_doc in VE gates + RoPE bases doubled.
