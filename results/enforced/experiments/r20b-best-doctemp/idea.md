# Doc-position term in the query temperature
Parent: r14c-chain3-attn-reads-attn
Type: exploit

The doc-position attention temperature gave -0.0043 on baseline (r19d). Add it to this parent's content-based
query temperature: q *= 1 + tanh(W x + c_h f_t), f_t = log(1 + t - doc_start)/log(T), c init 0.
Confirm: < parent by > 0.0005.
