# Best model + attention reads the previous MLP output
Parent: r11a-ms-mra
Type: exploit

Complete the "undiluted read" chain: attention input = norm(x) + gamma_l norm(m_{l-1}) (in addition to the
MLP reading the fresh attention output). Confirm: < r11a (0.94787) by > 0.0005.
