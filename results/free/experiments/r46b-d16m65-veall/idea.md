# proxy (r45d) with value embeddings on every layer

Parent: r45d. Memory is no longer binding (67GB); value embeddings are cheap in time and removing them cost +0.003 (r40d). Put them on all 16 layers (was alternating).
