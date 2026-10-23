# Document position feature on qfeat-docpos
Parent: r18b-scalargate-prevtok-qfeat
Type: exploit

Add the position-within-document input feature (x0 += log(1 + t - doc_start)/log(T) * w; r18d: -0.0021 on
baseline) to this parent. Cheap ([B,T] ops + one add). Confirm: < parent by > 0.001.
