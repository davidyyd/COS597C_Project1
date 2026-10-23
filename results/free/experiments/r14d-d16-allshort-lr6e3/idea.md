# Window 256 on every layer except the last (pattern S)

Parent: r13c-d16-win256-lr6e3. Explore (efficiency/architecture): only the final layer sees full context (plus x0/value-embedding paths); others local 256. Saves more attention time if quality holds.
