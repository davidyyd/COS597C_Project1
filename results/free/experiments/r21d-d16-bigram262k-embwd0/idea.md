# d16 bigram 262k, no weight decay on embedding tables

Parent: r19c-d16-bigram262k. Explore: decoupled WD shrinks every row of the embedding tables (wte, value embeds, bigram) each step even when the row is unused in the batch; for large sparse tables this may erase rare-row information. WD 0 for the embedding group.
