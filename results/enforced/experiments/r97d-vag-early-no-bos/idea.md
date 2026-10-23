# Isolated early layers without their document's BOS key (sink test)
Parent: r83a-vag-dm-ingate
Type: explore (mechanism)
In the document-masked early layers each document's BOS is the only universal key (a sink). Hide it (except for
the BOS query itself): if worse than r83a (0.94006), isolated layers rely on the BOS as a no-op sink.
