# Attention-to-attention undiluted read on the faster r11a
Parent: r11a-ms-mra
Type: exploit

r11a (0.94787, ~67 ms) has the MLP reading norm(a_l) and keeps prev-token keys. Add only the
attention <- norm(a_{l-1}) read (the last link added in r14c), giving a cheaper chain that keeps the
prev-token keys. Confirm: < r11a by > 0.001.
