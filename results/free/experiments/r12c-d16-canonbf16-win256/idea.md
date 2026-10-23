# Short sliding windows (256) for the S layers

Parent: r11b-d16-canon-bf16. Explore (efficiency/architecture). S layers used window 1024 (half context); 256 cuts their attention cost ~4x (modded-nanogpt uses short windows widely). Long layers (every 4th + last) keep full context. Speed+quality vs r11b (2495s, 0.79477).
