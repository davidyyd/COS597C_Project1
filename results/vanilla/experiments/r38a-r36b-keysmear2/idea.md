# r36b (no readout) + selective lag-2 key smear
Parent: r36b-r36a-noreadout (adds r31a's lag-2 key smear, under the same selective gate). Type: exploit.

The lag-2 key smear (keys encode their 2-token prefix) was worth ~0.0006 in r32b vs r32a; the readout ~0.0004
for 3.8 s. Trading the readout for lag-2 keys should be a net gain within the cap.
Confirm: < 0.933890 (r36a) within the cap. (Run noise in pointer models is ~5e-4.)
