# d16 (r30d): learnable exclusive self-attention (XSA)

Parent: r30d-d16-nosmear. From modded-nanogpt records 82/86 (Zhai 2025): remove the component of each head's attention output along the token's own (normalized) value vector, gated per head: y <- y - tanh(alpha_h) <y, v_hat> v_hat, alpha zero-init (identical at start).
