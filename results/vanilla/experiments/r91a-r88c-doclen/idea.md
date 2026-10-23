# r88c + document-length-aware attention temperature
Parent: r88c-r87b-dist-nometa. Type: exploit (combine with the r90d explore win: -0.0024 on the control at
zero time cost). q *= 1 + a_{l,h} log(1 + tokens since BOS)/log(T) in every attention layer, on top of the
per-(head,channel) query gain. Confirm: < 0.923063 within the cap.
