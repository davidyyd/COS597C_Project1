# Role-specific reads: the time-shifted read feeds only the keys (gate(3,7) variant of the best)
Parent: r54b-best-gate37 (= r53a with the low-rank gate in layers 3, 7)
Type: exploit

Keys read u + b2 norm(m_{l-1})[t-1] ("what came before me", as the prev-token key shortcut), queries and values
read u = norm(x) + b1 norm(m_{l-1})[t]. No extra matmuls. (On r53a it measured 69.1 ms.) Confirm: < r54b.
