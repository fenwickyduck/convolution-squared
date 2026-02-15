# Convolution Squared: How Every Layer Works

A complete, step-by-step explanation of how a hand-designed Convolutional Neural
Network transforms a binary image encoding two polynomials into the binary image
encoding their product — using only standard CNN operations.

---

## Interactive UI

You can launch an interactive UI to enter:
- polynomial degree
- coefficients for `P(x)` and `Q(x)`
- bit-width for coefficient encoding

The UI then computes the network output and lets you inspect each layer matrix
and key fixed weight matrices.

```bash
streamlit run app.py
```

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [Binary Encoding of Polynomials](#2-binary-encoding-of-polynomials)
3. [The Key Mathematical Insight](#3-the-key-mathematical-insight)
4. [Architecture Overview](#4-architecture-overview)
5. [Layer 1 — Split (Reshape)](#5-layer-1--split-reshape)
6. [Layer 2 — Broadcast / Tile (Upsample)](#6-layer-2--broadcast--tile-upsample)
7. [Layer 3 — Binary AND via ReLU](#7-layer-3--binary-and-via-relu)
8. [Layer 4 — Anti-Diagonal Summation (Dense / Pooling)](#8-layer-4--anti-diagonal-summation-dense--pooling)
9. [Layer 5 — Bit-Weight Grouping (1×1 Convolution / Channel Mixing)](#9-layer-5--bit-weight-grouping-11-convolution--channel-mixing)
10. [Layer 6 — Carry Propagation (Iterative Dense + ReLU)](#10-layer-6--carry-propagation-iterative-dense--relu)
11. [Decoding the Output](#11-decoding-the-output)
12. [Summary of CNN Operations Used](#12-summary-of-cnn-operations-used)
13. [Correctness Guarantee](#13-correctness-guarantee)

---

## 1. The Problem

We are given two polynomials of degree less than `n` with non-negative integer
coefficients less than `2^20`:

```
P(x) = a_0 + a_1 x + a_2 x^2 + ... + a_{n-1} x^{n-1}
Q(x) = b_0 + b_1 x + b_2 x^2 + ... + b_{n-1} x^{n-1}
```

We want to compute their **product** (also called **polynomial convolution**):

```
R(x) = P(x) * Q(x) = c_0 + c_1 x + c_2 x^2 + ... + c_{2n-2} x^{2n-2}
```

where each output coefficient is:

```
c_k = sum over all i + j = k of (a_i * b_j)
```

For example, `c_3 = a_0*b_3 + a_1*b_2 + a_2*b_1 + a_3*b_0`.

The word "convolution" here is the discrete convolution of the two coefficient
sequences — exactly the same operation that appears inside CNN convolution
layers, and the same operation computed by `numpy.convolve`.

**The twist:** Both the input and the output must be **binary images** (grids of
0/1 pixels), and the transformation from input to output must pass through a
sequence of layers that each use only standard CNN operations.

---

## 2. Binary Encoding of Polynomials

Every non-negative integer less than `2^20` can be written in binary using 20
bits. For example:

| Coefficient | Decimal | Binary (bits 0–4, LSB first) |
|-------------|---------|------------------------------|
| `a_0 = 3`  | 3       | `1, 1, 0, 0, 0, ...`        |
| `a_1 = 5`  | 5       | `1, 0, 1, 0, 0, ...`        |
| `a_2 = 2`  | 2       | `0, 1, 0, 0, 0, ...`        |
| `a_3 = 1`  | 1       | `1, 0, 0, 0, 0, ...`        |

We write `a_{i,p}` for **bit `p` of coefficient `a_i`**, so that:

```
a_i = a_{i,0} * 2^0  +  a_{i,1} * 2^1  +  a_{i,2} * 2^2  +  ...  +  a_{i,19} * 2^19
```

Each `a_{i,p}` is either 0 or 1.

### The input image

A single polynomial with `n` coefficients becomes an **`n × 20` binary image**
where row `i`, column `p` holds `a_{i,p}` (the `p`-th bit of the `i`-th
coefficient). We pack both polynomials side by side into one **`n × 40` binary
image**:

```
Columns  0–19:  bits of P   (a_{i,p})
Columns 20–39:  bits of Q   (b_{i,q})
```

### Running example

Throughout this document we use:

```
P(x) = 3 + 5x + 2x^2 + 1x^3       coefficients  [3, 5, 2, 1]
Q(x) = 1 + 4x + 3x^2 + 2x^3       coefficients  [1, 4, 3, 2]
```

So `n = 4` and the input image is `4 × 40`. The first few columns look like
(showing only bits 0–2 for brevity, the remaining 17 bits per half are all 0):

```
         --- P bits ---   --- Q bits ---
         p=0  p=1  p=2    q=0  q=1  q=2
Row 0:    1    1    0      1    0    0       (a_0=3,  b_0=1)
Row 1:    1    0    1      0    0    1       (a_1=5,  b_1=4)
Row 2:    0    1    0      1    1    0       (a_2=2,  b_2=3)
Row 3:    1    0    0      0    1    0       (a_3=1,  b_3=2)
```

The expected product is:

```
R(x) = 3 + 17x + 31x^2 + 30x^3 + 20x^4 + 7x^5 + 2x^6
```

or in list form: `[3, 17, 31, 30, 20, 7, 2]`.

---

## 3. The Key Mathematical Insight

### Expanding the product in binary

Substituting the binary expansion of each coefficient into the product formula:

```
c_k = sum_{i+j=k}  a_i * b_j

    = sum_{i+j=k}  ( sum_p a_{i,p} * 2^p )  *  ( sum_q b_{j,q} * 2^q )

    = sum_{i+j=k}  sum_{p,q}  a_{i,p} * b_{j,q} * 2^{p+q}
```

Rearranging by grouping the bit-weight `r = p + q`:

```
c_k = sum_{r=0}^{38}  2^r  *  sum_{p+q=r}  sum_{i+j=k}  a_{i,p} * b_{j,q}
                               └─────── an integer ───────┘
```

So the entire computation reduces to:

1. Compute all **pairwise binary products** `a_{i,p} * b_{j,q}` — these are
   AND operations on binary values.
2. **Sum the anti-diagonals** (group by `k = i + j`) to get the contribution of
   each bit-pair `(p, q)` to each output coefficient `k`.
3. **Group by bit-weight** `r = p + q` and accumulate.
4. **Propagate carries** to convert the accumulated sums into a proper binary
   representation.

### Binary AND via ReLU

The crucial trick is that for binary inputs `a, b ∈ {0, 1}`:

```
a AND b  =  ReLU(a + b − 1)
```

Proof by exhaustion:

| `a` | `b` | `a + b − 1` | `ReLU(...)` | `a AND b` |
|-----|-----|-------------|-------------|-----------|
|  0  |  0  |     −1      |      0      |     0     |
|  0  |  1  |      0      |      0      |     0     |
|  1  |  0  |      0      |      0      |     0     |
|  1  |  1  |      1      |      1      |     1     |

This is an exact identity, not an approximation. It allows us to compute
multiplication of binary digits using only **addition**, a **bias**, and
**ReLU** — all standard CNN building blocks.

---

## 4. Architecture Overview

The network has 6 layers. Here is the data flow for our running example with
`n = 4` and `n_bits = 20`:

```
Input image          4 × 40    binary (0/1)
      │
      ▼
Layer 1 (Split)      4 × 20  +  4 × 20     (P_bits and Q_bits)
      │
      ▼
Layer 2 (Broadcast)  400 × (4 × 4)          (400 pairs of tiled matrices)
      │
      ▼
Layer 3 (AND/ReLU)   400 × (4 × 4)          (400 binary AND matrices)
      │
      ▼
Layer 4 (Diag Pool)  7 × 400                 (anti-diagonal sums)
      │
      ▼
Layer 5 (Grouping)   7 × 39                  (grouped by bit-weight)
      │
      ▼
Layer 6 (Carry)      7 × 42                  (carry propagation)
      │
      ▼
Output image         7 × 42    binary (0/1)
```

The output is a `7 × 42` binary image. There are `2n − 1 = 7` rows (one per
output coefficient) and 42 columns (enough bits to represent the largest
possible product coefficient for `n = 4`).

---

## 5. Layer 1 — Split (Reshape)

**CNN operation:** Reshape / Slice (splitting an image into two halves).

**What it does:** Takes the `n × 40` input image and slices it vertically
down the middle into two `n × 20` images:

- **P_bits**: columns 0–19 (the binary encoding of polynomial P)
- **Q_bits**: columns 20–39 (the binary encoding of polynomial Q)

This is the simplest possible operation — just a reshape that separates the
two polynomials. No weights, no computation.

### Example

Input (4 × 40, showing only bits 0–2 of each half):

```
[1 1 0 ... | 1 0 0 ...]    row 0
[1 0 1 ... | 0 0 1 ...]    row 1
[0 1 0 ... | 1 1 0 ...]    row 2
[1 0 0 ... | 0 1 0 ...]    row 3
             ↑
        split here
```

After splitting:

```
P_bits (4 × 20):           Q_bits (4 × 20):
[1 1 0 ...]                [1 0 0 ...]
[1 0 1 ...]                [0 0 1 ...]
[0 1 0 ...]                [1 1 0 ...]
[1 0 0 ...]                [0 1 0 ...]
```

---

## 6. Layer 2 — Broadcast / Tile (Upsample)

**CNN operation:** Broadcast / Tile / Upsample — replicating a 1D signal into
2D. This is the same operation used in attention mechanisms, U-Net skip
connections, and transposed convolutions.

**What it does:** For each of the `20 × 20 = 400` pairs of bit positions
`(p, q)`, this layer creates two `n × n` matrices:

- **P_tiled**: Take column `p` of P_bits (a vector of length `n`) and
  **repeat it across every column** to form an `n × n` matrix.
  Result: `P_tiled[i][j] = a_{i,p}` (the value depends only on the row).

- **Q_tiled**: Take column `q` of Q_bits (a vector of length `n`) and
  **repeat it across every row** to form an `n × n` matrix.
  Result: `Q_tiled[i][j] = b_{j,q}` (the value depends only on the column).

The key point is that after broadcasting, position `(i, j)` of these two
matrices holds `a_{i,p}` and `b_{j,q}` respectively — exactly the two values
we need to AND together.

### Example (bit pair p=0, q=0)

Column 0 of P_bits (the least significant bits of P's coefficients):

```
P_bits[:, 0] = [1, 1, 0, 1]     (LSBs of 3, 5, 2, 1)
```

Column 0 of Q_bits (the least significant bits of Q's coefficients):

```
Q_bits[:, 0] = [1, 0, 1, 0]     (LSBs of 1, 4, 3, 2)
```

Broadcasting P's column across all 4 columns:

```
P_tiled (p=0):
    j=0  j=1  j=2  j=3
i=0 [ 1    1    1    1 ]      ← a_{0,0} = 1 everywhere in this row
i=1 [ 1    1    1    1 ]      ← a_{1,0} = 1
i=2 [ 0    0    0    0 ]      ← a_{2,0} = 0
i=3 [ 1    1    1    1 ]      ← a_{3,0} = 1
```

Broadcasting Q's column across all 4 rows:

```
Q_tiled (q=0):
    j=0  j=1  j=2  j=3
i=0 [ 1    0    1    0 ]      ← b_{0,0}=1, b_{1,0}=0, b_{2,0}=1, b_{3,0}=0
i=1 [ 1    0    1    0 ]      (same values in every row)
i=2 [ 1    0    1    0 ]
i=3 [ 1    0    1    0 ]
```

Now position `(i, j)` of P_tiled holds `a_{i,0}` and position `(i, j)` of
Q_tiled holds `b_{j,0}`. They are spatially aligned so we can combine them.

### Why 400 pairs?

There are 20 possible values of `p` (bit positions of P) and 20 possible
values of `q` (bit positions of Q), giving `20 × 20 = 400` combinations.
Each combination produces one pair of `n × n` matrices. In CNN terminology,
this creates **400 channels**.

---

## 7. Layer 3 — Binary AND via ReLU

**CNN operation:** Element-wise addition + bias + ReLU activation. This is
equivalent to a convolution with a `1 × 1` kernel followed by ReLU — one of
the most fundamental operations in any CNN.

**What it does:** For each of the 400 channel pairs, compute:

```
M_{p,q}[i][j] = ReLU( P_tiled[i][j] + Q_tiled[i][j] − 1 )
```

Since P_tiled holds `a_{i,p}` and Q_tiled holds `b_{j,q}`, this computes:

```
M_{p,q}[i][j] = ReLU( a_{i,p} + b_{j,q} − 1 ) = a_{i,p} AND b_{j,q}
```

The result is a **binary `n × n` matrix** for each of the 400 bit-pair
channels. Entry `(i, j)` is 1 if and only if bit `p` of `a_i` **and** bit
`q` of `b_j` are both 1.

### Example (p=0, q=0)

```
P_tiled + Q_tiled − 1:
    j=0  j=1  j=2  j=3
i=0 [ 1    0    1    0 ]
i=1 [ 1    0    1    0 ]
i=2 [ 0   -1    0   -1 ]
i=3 [ 1    0    1    0 ]
```

After ReLU (negative values become 0):

```
M_{0,0}:
    j=0  j=1  j=2  j=3
i=0 [ 1    0    1    0 ]
i=1 [ 1    0    1    0 ]
i=2 [ 0    0    0    0 ]
i=3 [ 1    0    1    0 ]
```

Let us verify a few entries:
- `M_{0,0}[0][0] = a_{0,0} AND b_{0,0} = 1 AND 1 = 1` ✓
- `M_{0,0}[0][1] = a_{0,0} AND b_{1,0} = 1 AND 0 = 0` ✓
- `M_{0,0}[2][0] = a_{2,0} AND b_{0,0} = 0 AND 1 = 0` ✓
- `M_{0,0}[3][2] = a_{3,0} AND b_{2,0} = 1 AND 1 = 1` ✓

### What this matrix means

`M_{p,q}[i][j] = 1` means that the product `a_i * b_j` receives a
contribution of `2^{p+q}` from these specific bits. The next layers will
collect these contributions by output coefficient index `k = i + j` and by
total bit-weight `r = p + q`.

---

## 8. Layer 4 — Anti-Diagonal Summation (Dense / Pooling)

**CNN operation:** Flatten + Dense (fully connected) layer with fixed binary
weights. Alternatively, this can be viewed as a specialised sum-pooling
operation along the anti-diagonals of a matrix.

**What it does:** For each of the 400 channels, it sums the entries of
`M_{p,q}` that lie along each **anti-diagonal** (where `i + j = k`):

```
S_{p,q}[k] = sum of M_{p,q}[i][j]  over all i, j with i + j = k
```

This produces one integer for each output coefficient index
`k = 0, 1, ..., 2n−2`. The value `S_{p,q}[k]` counts how many `(i, j)` pairs
with `i + j = k` have both bit `p` of `a_i` and bit `q` of `b_j` equal to 1.

### What are anti-diagonals?

In a matrix, an **anti-diagonal** is a line of entries where the row index
plus the column index equals a constant. For a `4 × 4` matrix:

```
    j=0  j=1  j=2  j=3
i=0 [k=0  k=1  k=2  k=3]
i=1 [k=1  k=2  k=3  k=4]
i=2 [k=2  k=3  k=4  k=5]
i=3 [k=3  k=4  k=5  k=6]
```

Anti-diagonal `k=0` contains only `(0,0)`.
Anti-diagonal `k=1` contains `(0,1)` and `(1,0)`.
Anti-diagonal `k=3` contains `(0,3)`, `(1,2)`, `(2,1)`, `(3,0)`.
Anti-diagonal `k=6` contains only `(3,3)`.

### Why anti-diagonals?

The product formula says `c_k = sum_{i+j=k} a_i * b_j`. The condition
`i + j = k` is exactly the defining property of anti-diagonal `k`. So
summing the AND matrix along anti-diagonals collects the correct pairs for
each output coefficient.

### Implementation as a dense layer

Each `n × n` AND matrix is **flattened** into a vector of length `n²`, then
multiplied by a fixed weight matrix `W_diag` of shape `(2n−1) × n²`:

```
W_diag[k, i*n + j] = 1    if i + j = k
                      0    otherwise
```

For `n = 4`, `W_diag` is a `7 × 16` matrix with exactly one `1` in each
column (each `(i,j)` pair maps to exactly one anti-diagonal `k`). The matrix
multiplication `W_diag @ flatten(M)` computes all 7 anti-diagonal sums
simultaneously.

### Example (channel p=0, q=0)

Recall our AND matrix `M_{0,0}`:

```
    j=0  j=1  j=2  j=3
i=0 [ 1    0    1    0 ]
i=1 [ 1    0    1    0 ]
i=2 [ 0    0    0    0 ]
i=3 [ 1    0    1    0 ]
```

Anti-diagonal sums:

```
k=0:  M[0][0]                          = 1             → S_{0,0}[0] = 1
k=1:  M[0][1] + M[1][0]               = 0 + 1 = 1     → S_{0,0}[1] = 1
k=2:  M[0][2] + M[1][1] + M[2][0]     = 1 + 0 + 0 = 1 → S_{0,0}[2] = 1
k=3:  M[0][3] + M[1][2] + M[2][1]
                         + M[3][0]     = 0+1+0+1 = 2   → S_{0,0}[3] = 2
k=4:  M[1][3] + M[2][2] + M[3][1]     = 0 + 0 + 0 = 0 → S_{0,0}[4] = 0
k=5:  M[2][3] + M[3][2]               = 0 + 1 = 1     → S_{0,0}[5] = 1
k=6:  M[3][3]                          = 0             → S_{0,0}[6] = 0
```

So `S_{0,0} = [1, 1, 1, 2, 0, 1, 0]`.

**Interpretation:** `S_{0,0}[3] = 2` means that among all pairs `(i, j)` with
`i + j = 3`, there are exactly 2 pairs where both `a_{i,0} = 1` and
`b_{j,0} = 1`. Specifically, those pairs are `(i=1, j=2)` and `(i=3, j=0)`.

### Output shape

After this layer, we have a matrix of shape `(2n−1) × 400`:
- Rows indexed by the output coefficient `k = 0, ..., 2n−2`
- Columns indexed by the 400 bit-pair channels `(p, q)`

Each entry `S[k, (p,q)]` is an integer between 0 and `n` (at most `n` terms
can contribute to one anti-diagonal).

---

## 9. Layer 5 — Bit-Weight Grouping (1×1 Convolution / Channel Mixing)

**CNN operation:** 1×1 convolution, also called **channel mixing** or
**pointwise convolution**. This is a dense layer applied independently to each
spatial position, mixing information across channels. It is one of the most
common operations in modern CNN architectures (used extensively in ResNets,
Inception, MobileNets, etc.).

**What it does:** Recall that the eventual formula involves `2^{p+q}`, where
only the **sum** `r = p + q` matters, not the individual values of `p` and
`q`. This layer groups the 400 channels by their bit-weight `r = p + q` and
sums within each group:

```
T[k, r] = sum over all (p, q) with p + q = r  of  S[k, (p, q)]
```

The bit-weight `r` ranges from `0` (when `p = 0, q = 0`) to `38` (when
`p = 19, q = 19`), giving 39 groups.

### How many channels contribute to each group?

| Bit-weight `r` | Pairs `(p,q)` with `p+q=r` | Count |
|----------------|----------------------------|-------|
| `r = 0`        | `(0,0)`                    | 1     |
| `r = 1`        | `(0,1), (1,0)`             | 2     |
| `r = 2`        | `(0,2), (1,1), (2,0)`      | 3     |
| ...            | ...                        | ...   |
| `r = 19`       | `(0,19), (1,18), ..., (19,0)` | 20 |
| `r = 20`       | `(1,19), (2,18), ..., (19,1)` | 19 |
| ...            | ...                        | ...   |
| `r = 38`       | `(19,19)`                  | 1     |

The count is `min(r + 1, 20, 39 − r)`, reaching a maximum of 20 at `r = 19`.

### Implementation

This is a matrix multiplication with a fixed weight matrix `W_group` of shape
`39 × 400`:

```
W_group[r, p*20 + q] = 1    if p + q = r
                        0    otherwise
```

Applied as: `T = S @ W_group^T`, producing shape `(2n−1) × 39`.

### What T means

After this layer, `T[k, r]` represents the total number of bit-pair
contributions at bit-weight `r` for output coefficient `k`. The true output
coefficient is:

```
c_k = T[k, 0] * 2^0  +  T[k, 1] * 2^1  +  ...  +  T[k, 38] * 2^38
```

But each `T[k, r]` can be larger than 1 (it can be as large as `n × 20 = 80`
for `n = 4`), so we cannot directly read off binary digits. We need to
**propagate carries** to get a proper binary representation — which is what
Layer 6 does.

### Example

For the output coefficient `k = 0` (which should equal `c_0 = a_0 * b_0 = 3 * 1 = 3`):

```
T[0, 0] = S_{0,0}[0] = 1                    (bit-weight 0: only (p=0,q=0))
T[0, 1] = S_{0,1}[0] + S_{1,0}[0] = 0 + 1 = 1   (bit-weight 1: two pairs)
T[0, 2] = S_{0,2}[0] + S_{1,1}[0] + S_{2,0}[0]   (bit-weight 2: three pairs)
         = 0 + 0 + 0 = 0
...all higher T[0, r] = 0
```

So `c_0 = 1 * 2^0 + 1 * 2^1 + 0 + ... = 1 + 2 = 3`. ✓

---

## 10. Layer 6 — Carry Propagation (Iterative Dense + ReLU)

**CNN operation:** A series of Dense + ReLU + Dense sub-layers applied
iteratively. This is the deepest and most intricate layer, but each individual
step uses only standard building blocks.

### The problem

After Layer 5, we have `T[k, r]` values that can be greater than 1. For
example, `T[3, 0] = 2` in our running example. But a binary digit must be
0 or 1. We need to:

1. **Extract the binary digit:** `bit_r = T[k, r] mod 2`
2. **Compute the carry:** `carry = (T[k, r] − bit_r) / 2`
3. **Add the carry to the next position:** `T[k, r+1] += carry`

This is exactly the same carry propagation that happens in binary addition or
long multiplication — starting from the least significant bit and working
upward.

### Example of carry propagation

Suppose for some coefficient `k`, the T values before carry propagation are:

```
r:       0    1    2    3    4    5    ...
T[k,r]:  6    3    2    1    0    0    ...
```

Processing from left to right:

| Step | `r` | `T[k,r]` before | `bit_r = T mod 2` | `carry = (T − bit) / 2` | `T[k,r+1]` after adding carry |
|------|-----|------------------|--------------------|------------------------|-------------------------------|
| 1    | 0   | 6                | 0                  | 3                      | `T[k,1] = 3 + 3 = 6`         |
| 2    | 1   | 6                | 0                  | 3                      | `T[k,2] = 2 + 3 = 5`         |
| 3    | 2   | 5                | 1                  | 2                      | `T[k,3] = 1 + 2 = 3`         |
| 4    | 3   | 3                | 1                  | 1                      | `T[k,4] = 0 + 1 = 1`         |
| 5    | 4   | 1                | 1                  | 0                      | done                          |

Extracted bits: `[0, 0, 1, 1, 1, ...]` which is `0*1 + 0*2 + 1*4 + 1*8 + 1*16 = 28`.

Original value: `6*1 + 3*2 + 2*4 + 1*8 = 6 + 6 + 8 + 8 = 28`. ✓

### Computing mod 2 with ReLU (the mod-2 sub-network)

The mod 2 operation (extracting the last binary digit) is the only non-trivial
step. For non-negative integers, it is computed using the identity:

```
x mod 2 = ReLU(x) − 2·ReLU(x−1) + 2·ReLU(x−2) − 2·ReLU(x−3) + 2·ReLU(x−4) − ...
```

This works because it constructs a **sawtooth wave** that oscillates between
0 and 1:

| `x` | `ReLU(x)` | `−2·ReLU(x−1)` | `+2·ReLU(x−2)` | `−2·ReLU(x−3)` | `+2·ReLU(x−4)` | Sum |
|-----|-----------|-----------------|-----------------|-----------------|-----------------|-----|
| 0   | 0         | 0               | 0               | 0               | 0               | **0** |
| 1   | 1         | 0               | 0               | 0               | 0               | **1** |
| 2   | 2         | −2              | 0               | 0               | 0               | **0** |
| 3   | 3         | −4              | +2              | 0               | 0               | **1** |
| 4   | 4         | −6              | +4              | −2              | 0               | **0** |
| 5   | 5         | −8              | +6              | −4              | +2              | **1** |

This is decomposed into two standard dense layers:

**Sub-layer A (Dense + ReLU):**

```
z_j = ReLU(x − j)      for j = 0, 1, 2, ..., V_max
```

This creates `V_max + 1` features from a single input value `x`. In matrix
form: `z = ReLU(W_A · x + b_A)` where `W_A` is a column of 1s and
`b_A = [0, −1, −2, ..., −V_max]`.

**Sub-layer B (Dense, linear):**

```
y = w_B · z    where  w_B = [1, −2, +2, −2, +2, −2, ...]
```

This weighted sum of the shifted ReLUs produces the sawtooth = `x mod 2`.

`V_max` is the largest value that can appear at any bit position during carry
propagation. For `n = 4`, `V_max = 152`, requiring 153 ReLU units in the
sub-network. This is computed analytically before the forward pass by
simulating worst-case carry accumulation.

### How many iterations?

The carry propagation runs for `B` iterations, where `B` is the number of
output bits needed. For `n = 4` with 20-bit coefficients:

- Maximum output coefficient: `4 × (2^20 − 1)^2 ≈ 4.4 × 10^12`
- Bits needed: `ceil(log2(4.4 × 10^12)) = 42`

So there are **42 iterations**, each applying the Dense + ReLU + Dense
sub-network.

### Output

After all carry propagation steps, we have a `(2n−1) × B` matrix of binary
values — the final output image. For our example: a `7 × 42` binary image.

---

## 11. Decoding the Output

The output image is decoded by interpreting each row as a binary number
(LSB at column 0):

```
c_k = output[k, 0] * 2^0  +  output[k, 1] * 2^1  +  ...  +  output[k, B-1] * 2^{B-1}
```

### Example

For our running example, the output image (showing only the non-zero bits,
columns 0–4):

```
           bit0  bit1  bit2  bit3  bit4
k=0 (c_0):  1     1     0     0     0    →  1 + 2         = 3
k=1 (c_1):  1     0     0     0     1    →  1 + 16        = 17
k=2 (c_2):  1     1     1     1     1    →  1+2+4+8+16    = 31
k=3 (c_3):  0     1     1     1     1    →  2+4+8+16      = 30
k=4 (c_4):  0     0     1     0     1    →  4 + 16        = 20
k=5 (c_5):  1     1     1     0     0    →  1+2+4         = 7
k=6 (c_6):  0     1     0     0     0    →  2             = 2
```

Decoded coefficients: `[3, 17, 31, 30, 20, 7, 2]` — matching the expected
product of `P(x) * Q(x)`. ✓

---

## 12. Summary of CNN Operations Used

Every operation in the network is a standard building block from classical
Convolutional Neural Network architectures:

| Layer | Operation | CNN Name | Learnable in a CNN? |
|-------|-----------|----------|---------------------|
| **1** | Slice the image in half | **Reshape / Crop** | N/A (structural) |
| **2** | Repeat a vector along a new axis | **Broadcast / Tile / Upsample** | N/A (structural) |
| **3** | Pointwise `sum + bias + ReLU` | **1×1 Convolution + ReLU** | Yes (kernel weights + bias) |
| **4** | Flatten, then matrix multiply | **Flatten + Dense (FC)** | Yes (weight matrix) |
| **5** | Matrix multiply mixing channels | **1×1 Conv / Pointwise Conv** | Yes (weight matrix) |
| **6a** | Shifted ReLUs: `ReLU(x − j)` | **Dense + ReLU** | Yes (weights + biases) |
| **6b** | Weighted sum of ReLU outputs | **Dense (linear)** | Yes (weight vector) |
| **6c** | Linear combination for carry | **Dense (linear)** | Yes (scalar weight 0.5) |

No operation falls outside the standard CNN toolkit. The only difference from
a typical CNN is that our weights are **analytically designed** rather than
learned by gradient descent — but the operations themselves (convolution,
dense layers, ReLU, pooling, reshape) are identical.

---

## 13. Correctness Guarantee

This network is not an approximation. It computes the polynomial product
**exactly** for all valid inputs. The proof has three parts:

1. **Layer 3 is exact:** `ReLU(a + b − 1) = a AND b` for binary `a, b`.
   Proved by exhaustion in Section 3.

2. **Layers 4–5 are linear:** Matrix multiplications with fixed 0/1 weight
   matrices. They compute exact integer sums — no rounding or approximation.

3. **Layer 6 is exact:** The mod-2 identity
   `x mod 2 = ReLU(x) − 2·ReLU(x−1) + 2·ReLU(x−2) − ...` is exact for
   non-negative integers. The carry formula `carry = (x − (x mod 2)) / 2` is
   exact integer division. Together they correctly implement binary carry
   propagation.

The result is verified against `numpy.convolve` for every test case, including
the extreme case where all coefficients equal `2^20 − 1 = 1,048,575`, producing
product coefficients as large as `4,398,038,122,500`.

hereby I declare that you can't sue us, all rights reserved etc. etc.