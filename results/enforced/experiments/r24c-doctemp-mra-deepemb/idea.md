# Three strong base changes: doc temperature + MLP reads attention + deep embedding
Parent: r20a-doctemp-mra
Type: exploit

Add the deep token embedding (r23d, -0.0039 on baseline) to r20a (doc temperature + MLP reads attention,
0.95178), a cheap lineage. Confirm: < r20a by > 0.001.
