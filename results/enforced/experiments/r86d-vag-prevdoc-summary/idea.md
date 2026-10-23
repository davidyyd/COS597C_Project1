# Late layers read the previous document's final state (neighbour-document summary)
Parent: r83a-vag-dm-ingate
Type: explore
Hypothesis: late layers benefit from cross-document attention, and the neighbouring document is all they need
(r84c). The residual state at the previous document's last token is a causal summary of that document; a direct,
undiluted read of it (one scalar per layer) may give late layers the neighbour context more cheaply than attention.
Change: layers 4-7: attention input u = norm(x) + c_l norm(x[start_t - 1]) (zero for the row's first document).
Confirm: below r83a (0.94006) by > 0.0005.
