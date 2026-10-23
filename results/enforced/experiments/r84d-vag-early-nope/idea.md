# No RoPE in the document-isolated early layers
Parent: r83a-vag-dm-ingate
Type: explore
Hypothesis: in the early, document-masked layers the causal mask from the document start already gives position
information (NoPE works with causal masks), and position-free matching may suit lexical early processing.
Change: layers 0-2 skip the rotary embedding. Confirm: below r83a (0.94006) by > 0.0005.
Closest work: NoPE; iRoPE (Llama 4: NoPE in global layers). Here NoPE in the document-isolated local layers.
