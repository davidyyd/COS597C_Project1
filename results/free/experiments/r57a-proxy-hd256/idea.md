# proxy: 7 heads of 256 (instead of 14 of 128)

Parent: r45d (0.76588). 28x64 was worse (+0.0024, r31d); try the other direction: 7 heads x 256, same params. Attention scale 0.12*sqrt(128/256)... keep logit range: scale 0.085.
