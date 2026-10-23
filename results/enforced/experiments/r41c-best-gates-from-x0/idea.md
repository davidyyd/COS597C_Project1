# Value gates computed from the token embedding
Parent: r38a-gated-proj
Type: exploit

The value-path gates (how much context vs token identity a value carries) read the block input x. Let them read
the token's (deep) embedding x0 instead: whether a token's value should carry its identity is largely a property
of the token. Same cost. Confirm: < r38a by > 0.0003.
