# Document position feature on scalargate-docpos
Parent: r18a-scalargate-prevtok
Type: exploit

Add the position-within-document input feature (x0 += log(1 + t - doc_start)/log(T) * w; r18d: -0.0021 on
baseline) to this parent. Cheap ([B,T] ops + one add). Confirm: < parent by > 0.001.
