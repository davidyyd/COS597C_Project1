# Document-jump RoPE: other documents are "far away"
Parent: r81c-vag-dm012-doctemp
Type: explore

Hypothesis: hard document masking helps early layers but hurts late ones, which use other documents' content.
A soft separation that keeps content matching but removes fine positional matching across documents may suit
all layers: advance the RoPE position by T at every document start. Inside a document relative positions are
unchanged; a key from another document looks >= T tokens away, so high-frequency rotary channels decorrelate
while slow channels (content) still match.
Change: RoPE position p_t = t + T * doc_index_t (doc_index = number of BOS tokens before t). No parameters.
Confirm: below r81c (0.94189) by > 0.0005.
Closest work: position-ID reset per document (known; breaks relative distances across documents), document
masking; here positions jump forward so cross-document distances are large but consistent.
