# r26a + multi-depth readout (bf16)
Parent: r26a-local0-vsmear-bf16 (adds r10d's depth readout). Type: exploit.

The depth readout helped (-0.0043 alone, -0.0023 in r12c) but was dropped for time; with the bf16
residual (h's are bf16) it is cheaper. Confirm: < r26a within the cap.
