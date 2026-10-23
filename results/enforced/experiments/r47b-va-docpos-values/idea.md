# Values carry the document position (normalized bucket table)
Parent: r43a-base-valuepath
Type: explore

Hypothesis: attention outputs say what was retrieved but not from where; with the doc-position effects seen
earlier (doc temperature, -0.0043), knowing whether the attended tokens are near the document start could help.
Each layer has a 16x512 table of log2 document-position buckets, normalized at use and mixed into the values with
a signed per-head gate (0 at init): v <- v + tanh(W x) norm(PE[bucket(t - doc_start)]).
Confirm: < r43a (0.94690) by > 0.001.
