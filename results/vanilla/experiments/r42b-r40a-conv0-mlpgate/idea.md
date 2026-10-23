# r40a with layer 0 back to the value conv (cheaper) + grouped MLP hidden gate
Parent: r40a-r39b-noptr-noreadout (adds r41d's grouped MLP gate; layer 0 as in r18b). Type: exploit.
Local attention in layer 0 cost ~5 s for ~0.001 over the conv; the grouped MLP gate gave -0.0024 on the
control for ~5 s. Swapping should be a net gain within the cap.
Confirm: < 0.932561 within the cap.
