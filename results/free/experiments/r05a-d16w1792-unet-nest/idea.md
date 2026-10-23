# Wider-shallower: d16w1792 (similar params to d24w1408)

Parent: r04b-d24w1408-lr5e3-unet-nest. Exploit (shape): 16 layers x 1792 (14 heads of 128), 616M block params vs 571M. Fewer, larger matmuls -> higher MFU; tests width vs depth at fixed wall clock.
