# Partial RoPE: the lowest-frequency half of the channel pairs is not rotated
Parent: r25a-bf16-resid. Type: explore (known method, tested as a base option).

Hypothesis: the lowest RoPE frequencies barely rotate over 2048 tokens and mostly add slow positional
drift to content matching; leaving them unrotated gives clean position-free channels.
Confirm: < 0.95726. Closest work: GPT-NeoX partial rotary, modded-nanogpt half-truncated RoPE.
