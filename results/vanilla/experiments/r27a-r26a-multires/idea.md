# r26a + multi-resolution heads in the sliding-window layers
Parent: r26a-local0-vsmear-bf16 (adds r26d). Type: exploit.

Multi-resolution heads (2 of 4 heads with a 128 window in each 1024-window layer) were both better
(-0.002) and faster (-6.6 s) on the control. In the stack they should add and free time for the readout.
Confirm: < 0.935998 and faster than 256.8 s.
