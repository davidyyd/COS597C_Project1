# Key-only shift model + attention reads the previous attention output
Parent: r54a-gate37-k-shift-only (0.93555)
Type: exploit

Add back the attention <- norm(a_{l-1})[t] read (all of q, k, v). Confirm: < r54a.
