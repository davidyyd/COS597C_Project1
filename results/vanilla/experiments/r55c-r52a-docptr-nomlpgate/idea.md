# r55a with the pointer restricted to the current document
Parent: r55a-r52a-nomlpgate. Type: exploit.
Doc masking helped only together with the pointer; the pointer itself still copies successors from earlier
documents in the packed row. Restricting its candidate positions to the same document removes spurious
copies (and is consistent with the masked attention). Confirm: < r55a within the cap.
