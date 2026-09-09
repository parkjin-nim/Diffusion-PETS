# Aggregation definition

FULL = A* + R_C; FIXED_ANCHOR = A*; ANTI_ANCHOR = A* + P_A*(R_C).

The identical shared `halfspace_projection` is used at encoder and decoder. Direct harmonic tolerance is 0.05; no transitive closure.

Frozen Anti-Anchor inference is not equivalent to Anti-Anchor-aware training.
