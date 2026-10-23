# Best model + LM head reads the last MLP output
Parent: r11a-ms-mra
Type: exploit

Add the head's undiluted read of the last MLP output (x_head = norm(x_L) + beta norm(m_L); r11d: -0.0014
on baseline) to the best model r11a (0.94787). Nearly free in time. Confirm: < 0.9474.
