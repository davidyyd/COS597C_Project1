# Value-embedding gate reads the whole input
Parent: r25a-cheap-de-lowrank-half
Type: exploit

The baseline's value-embedding gate is computed from the first 32 channels of the normalized input only.
Let it read all 512 (W [4, 512], init 0). Cheap. Confirm: < r25a (0.94441) by > 0.0005.
