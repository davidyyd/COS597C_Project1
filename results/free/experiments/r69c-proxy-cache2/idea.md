# proxy backin5-mix + second cache: layers 4-8 attention read (mix of) the layer-3 activation

Parent: r67a-proxy-backin5-mix (0.76388). The cached-input trick helped for the last layers; try it for the middle block too: cache x after layer 3, attention of layers 4-8 reads x_c1 + m*(x - x_c1) (m learnable per block, init 0).
