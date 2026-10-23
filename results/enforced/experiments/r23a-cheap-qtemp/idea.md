# Cheap lineage + content term in the query temperature
Parent: r22b-cheap-ms-x0gate
Type: exploit

r22b (0.94672, 64.5 ms) has only the doc-position temperature. Add the content term:
q *= 1 + tanh(W x + c_h f_doc) (as in the best model). Confirm: < r22b by > 0.0005.
