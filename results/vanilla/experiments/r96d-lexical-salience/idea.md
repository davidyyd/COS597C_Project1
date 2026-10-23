# Lexical (token-identity) key/value salience
Parent: r95d-key-value-salience. Type: explore (is salience lexical or contextual?).
Same as r95d but the per-key [salience | value weight] come from a per-layer table indexed by the token id
x_s (8192 x 8, zero-init) instead of a projection of the contextual state. If it matches r95d (0.952950),
salience is mostly a token property (stopwords vs content); if worse, context matters.
