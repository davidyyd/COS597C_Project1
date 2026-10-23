# r79a + selective MLP token shift in layers 4-7
Parent: r79a-r78a-dropve3. Type: exploit (combine with the r80d explore win, -0.0023 on the control).
Head-to-head with r80a (MLP gate in layers 4-7, 0.923610): the same layers get the selective token shift
h' = h + s_t mu (h_{t-1} - h_t) instead. Confirm: < 0.923610 within the cap.
