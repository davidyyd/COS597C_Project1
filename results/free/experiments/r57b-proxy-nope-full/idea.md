# proxy: no RoPE on the full-attention layers (RoPE only on sliding-window layers)

Parent: r45d. Explore (iRoPE-style, Llama 4): global layers without positional rotation can use content-based long-range retrieval; local layers keep (partial) RoPE; causal masking still gives order info.
