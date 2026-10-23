# Early layers fully document-isolated: time-shifted reads also stop at document starts
Parent: r81a-dm012-lr357
Type: exploit
Hypothesis: early layers want to see only the current document (doc masking in 0-2 helps; later layers want
cross-document access). In r81a the early layers still receive the previous position's sublayer outputs through
the time-shifted reads, which at a document's first token come from the previous document. Zeroing those reads
at document starts in layers 0-2 completes the isolation (r53c did this in all layers: no gain, but there the
late layers were not meant to be isolated).
Confirm: below r81a (0.93000) by > 0.0003.
