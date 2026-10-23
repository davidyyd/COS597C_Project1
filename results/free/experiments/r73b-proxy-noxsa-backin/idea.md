# proxy backin5-mix: XSA disabled in the 5 backout-input layers

Parent: r67a. Those layers' attention reads a decoupled input; check whether XSA still helps there (modded-nanogpt found XSA worse on some layers). xsa factor 0 for layers 11-15 (alpha kept, zero effect).
