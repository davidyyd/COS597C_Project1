# r49b + token metadata also in the value embeddings
Parent: r49b-lean-tokmeta-local0. Type: exploit.
Value embeddings inject token identity into attention values; adding a per-table metadata embedding
(orthographic class) lets values carry those shared features too (cheap: 4 small tables).
Confirm: < 0.931007 within the cap.
