# Local-layer document masking that keeps the row's first BOS visible (sink)
Parent: r78d-vag-local-docmask
Type: explore (mechanism)
Question: masking the late local layers hurts (r79d). Is that because masking removes a shared attention sink
(the row's first BOS), which late layers use to "attend to nothing"? Here all local layers are doc-masked but
position 0 stays visible. If this beats r78d (0.94428) and approaches r79c (early-only, 0.94340), the late-layer
harm is about the sink; if not, late layers really use other documents' content.
