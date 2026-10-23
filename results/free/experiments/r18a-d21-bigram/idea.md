# d21 + hashed bigram embedding (65521 x 1792) added to the token embedding

Parent: r17a-d21-b16-SL-wu2 (best, 0.78146). Explore at scale: the 8192-token vocab is small; a hashed (previous, current) token-pair embedding (recent modded-nanogpt records use a bigram hash embedding) adds lexical capacity almost for free (one lookup; 117M Adam params at the embedding LR). Zero init; x = norm(wte(t) + bigram(hash(t-1, t))).
