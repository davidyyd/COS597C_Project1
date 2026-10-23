# Token-identity offset for the query temperature
Parent: r05b-best-x0gate
Type: exploit

Query temperature = 1 + tanh(W x + e[token, layer, head]) where e is a learned table (8192 x 32, init 0,
embedding). Some tokens (e.g. closing brackets, repeated names) should always query sharply; a table learns
this directly instead of through W x from a contextual state. Confirm: < r05b by > 0.0005.
