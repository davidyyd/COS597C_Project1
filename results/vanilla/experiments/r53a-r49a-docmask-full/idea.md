# r49a + document masking in the two full-attention layers
Parent: r49a-lean-tokmeta (adds r52a's doc masking; drops r51b's pointer to fit the cap). Type: exploit.
Doc masking in the global layers gave -0.0014 on r51b (r52a) but ran 265 s; r49a has ~14 s headroom.
Confirm: < 0.930749 within the cap.
