# r30a + per-layer mix between the contextual and the plain token embedding in the x0 re-injection
Parent: r30a-r29a-mlpsmear. Type: exploit.

The re-injected x0 is the contextual (lag-conv) embedding at every layer. Late layers may want the pure
token identity, early layers local context; a per-layer scalar mix (init 0 = current behaviour) lets
each layer choose. Confirm: < r30a within the cap.
