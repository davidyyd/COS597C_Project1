# Cheap composition + head self-gate (scalar + low-rank in odd layers)
Parent: r50c-compA-fullchain
Type: exploit

r50c dropped the head self-gate; re-measured on r43a the scalar+low-rank gate gave -0.0022. Add it (low-rank part
in odd layers only, for time). Confirm: < r50c by > 0.0005.
