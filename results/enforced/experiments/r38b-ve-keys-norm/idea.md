# Normalized value embeddings into keys
Parent: r37c-vesep-noouter
Type: exploit

Value embeddings into keys gave -0.0014 on baseline but nothing on later models; with normalized mixing (the
key gets k <- norm(k) + tanh(W x) norm(ve) before RoPE / QK-norm) the balance is controlled. Confirm: < r37c.
