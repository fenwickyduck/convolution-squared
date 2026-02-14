"""
Convoluted: CNN-Based Polynomial Convolution
=============================================

A hand-designed Convolutional Neural Network that transforms a binary image
encoding two polynomials into the binary image encoding their product.

All network weights are analytically computed (no training). Each layer uses
only standard CNN building blocks: reshape, broadcast/tile (upsample),
dense (fully connected), ReLU activation, and sum pooling.

Architecture
------------
    Input:  n x 40 binary image (two polynomials, 20 bits each)
    Layer 1 (Split):     Reshape into P_bits (n x 20) and Q_bits (n x 20)
    Layer 2 (Broadcast): Tile to create 400 pairs of n x n matrices
    Layer 3 (AND/ReLU):  Binary AND via ReLU(sum - 1) -> 400 n x n binary matrices
    Layer 4 (Diag Pool): Anti-diagonal summation via dense layer -> (2n-1) x 400
    Layer 5 (Group):     Bit-weight channel mixing via 1x1 conv  -> (2n-1) x 39
    Layer 6 (Carry):     Iterative Dense+ReLU for binary carry propagation
    Output: (2n-1) x B binary image (product polynomial)

Mathematical basis
------------------
For polynomials P = sum a_i x^i, Q = sum b_j x^j, their product R = P*Q has
coefficients c_k = sum_{i+j=k} a_i * b_j.

Expanding in binary (a_i = sum_p a_{i,p} * 2^p):

    c_k = sum_{i+j=k} sum_{p,q} a_{i,p} * b_{k-i,q} * 2^{p+q}

The key insight is that binary AND can be computed via ReLU:
    a AND b = ReLU(a + b - 1)   for a, b in {0, 1}

This allows bilinear products through standard CNN operations.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")                   # non-interactive backend (save only)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ================================================================
#  Part 1 — CNN Building Blocks
# ================================================================

def relu(x):
    """ReLU activation: element-wise max(0, x)."""
    return np.maximum(0, x)


def dense(x, W, b=None):
    """Fully connected (dense) layer: y = x W^T + b.

    Args:
        x: Input array, shape (..., d_in).
        W: Weight matrix, shape (d_out, d_in).
        b: Optional bias vector, shape (d_out,).
    Returns:
        Output array, shape (..., d_out).
    """
    y = x @ W.T
    if b is not None:
        y = y + b
    return y


def broadcast_to_matrix(vec, size, axis):
    """Broadcast a 1-D vector into a 2-D matrix (tile / upsample).

    Args:
        vec:  1-D array of length n.
        size: Size of the new dimension.
        axis: 0 -> repeat across columns: result[i, j] = vec[i]
              1 -> repeat across rows:    result[i, j] = vec[j]
    Returns:
        2-D array of shape (len(vec), size) if axis==0,
        or (size, len(vec)) if axis==1.
    """
    if axis == 0:
        return np.tile(vec.reshape(-1, 1), (1, size))
    else:
        return np.tile(vec.reshape(1, -1), (size, 1))


def sum_pool(x, axis):
    """Global sum-pooling along the given axis."""
    return np.sum(x, axis=axis)


# ================================================================
#  Part 2 — Encoding / Decoding
# ================================================================

def encode_polynomial(coeffs, n_bits=20):
    """Encode polynomial coefficients as a binary image (n x n_bits).

    Row i, column p = bit p of coefficient a_i  (LSB at column 0).

    Args:
        coeffs: Sequence of non-negative integer coefficients.
        n_bits: Number of bits per coefficient (default 20).
    Returns:
        Binary (0/1) numpy array of shape (n, n_bits), dtype float64.
    """
    n = len(coeffs)
    image = np.zeros((n, n_bits), dtype=np.float64)
    for i, c in enumerate(coeffs):
        for p in range(n_bits):
            image[i, p] = float((int(c) >> p) & 1)
    return image


def decode_polynomial(binary_image):
    """Decode a binary image back to integer polynomial coefficients.

    Args:
        binary_image: Array of shape (m, B) with 0/1 entries.
    Returns:
        1-D int64 array of length m.
    """
    _, B = binary_image.shape
    powers = np.int64(2) ** np.arange(B, dtype=np.int64)
    return np.round(binary_image).astype(np.int64) @ powers


def encode_input(P_coeffs, Q_coeffs, n_bits=20):
    """Pack two polynomials into a single n x (2 * n_bits) binary image.

    Left  half (cols 0 .. n_bits-1):       P coefficient bits.
    Right half (cols n_bits .. 2*n_bits-1): Q coefficient bits.
    """
    assert len(P_coeffs) == len(Q_coeffs), \
        "Polynomials must have the same degree bound n"
    return np.hstack([
        encode_polynomial(P_coeffs, n_bits),
        encode_polynomial(Q_coeffs, n_bits),
    ])


# ================================================================
#  Part 3 — Polynomial Convolution CNN
# ================================================================

class PolynomialConvCNN:
    """Hand-designed CNN that computes polynomial multiplication.

    Every weight matrix is computed analytically for a given (n, n_bits).
    The forward pass is *exact* (not an approximation) for all valid inputs
    (non-negative integer coefficients < 2^n_bits).
    """

    def __init__(self, n, n_bits=20):
        """
        Args:
            n:      Degree bound — each polynomial has n coefficients.
            n_bits: Bits per coefficient (default 20, so coeffs < 2^20).
        """
        self.n = n
        self.n_bits = n_bits
        self.out_len = 2 * n - 1          # number of output coefficients
        self.n_pairs = n_bits * n_bits     # 400 bit-position pairs

        self._build_all_weights()

    # ----------------------------------------------------------------
    #  Weight construction
    # ----------------------------------------------------------------

    def _build_all_weights(self):
        """Pre-compute fixed weight matrices for every layer."""
        n, nb = self.n, self.n_bits

        # -- Layer 4: Anti-diagonal summation --
        # W_diag[k, i*n + j] = 1  iff  i + j == k
        self.W_diag = np.zeros((self.out_len, n * n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                self.W_diag[i + j, i * n + j] = 1.0

        # -- Layer 5: Bit-weight grouping --
        # W_group[r, p*nb + q] = 1  iff  p + q == r
        self.max_r = 2 * nb - 2                # 38 for nb=20
        self.W_group = np.zeros((self.max_r + 1, nb * nb), dtype=np.float64)
        for p in range(nb):
            for q in range(nb):
                self.W_group[p + q, p * nb + q] = 1.0

        # -- Carry-propagation bounds --
        self._compute_carry_bounds()

        # -- Layer 6: Mod-2 ReLU network weights --
        V = self.global_v_max
        #   Sub-layer A:  z_j = ReLU(x - j)
        #     W_A is (V+1, 1) of ones,  b_A = [0, -1, -2, …, -V]
        self.W_mod_A = np.ones((V + 1, 1), dtype=np.float64)
        self.b_mod_A = -np.arange(V + 1, dtype=np.float64)

        #   Sub-layer B:  y = w_B · z  with alternating signs
        #     mod2(x) = ReLU(x) - 2·ReLU(x-1) + 2·ReLU(x-2) - …
        self.w_mod_B = np.zeros(V + 1, dtype=np.float64)
        self.w_mod_B[0] = 1.0
        for j in range(1, V + 1):
            self.w_mod_B[j] = 2.0 if j % 2 == 0 else -2.0

    def _compute_carry_bounds(self):
        """Determine output bit-width and max intermediate carry values."""
        n, nb = self.n, self.n_bits

        # Number of (p, q) pairs with p + q = r
        pair_counts = np.zeros(self.max_r + 1, dtype=np.int64)
        for r in range(self.max_r + 1):
            pair_counts[r] = min(r + 1, nb, 2 * nb - 1 - r)

        # Simulate worst-case carry propagation
        headroom = self.max_r + 1 + 60      # generous extra bits
        v_max = np.zeros(headroom, dtype=np.int64)
        carry = 0
        last_nonzero = 0
        for r in range(headroom):
            init = n * pair_counts[r] if r <= self.max_r else 0
            v_max[r] = init + carry
            if v_max[r] > 0:
                last_nonzero = r
            carry = v_max[r] // 2
            if init == 0 and carry == 0:
                break

        self.output_bits = last_nonzero + 1
        self.v_max = v_max[:self.output_bits]
        self.global_v_max = int(np.max(self.v_max))

    # ----------------------------------------------------------------
    #  Mod-2 via Dense + ReLU + Dense
    # ----------------------------------------------------------------

    def _mod2_relu(self, x):
        """Compute x mod 2 for non-negative integers via two dense layers.

        For binary digits a_{i,p}:
            mod2(x) = ReLU(x)  - 2·ReLU(x-1) + 2·ReLU(x-2) - 2·ReLU(x-3) + …

        Decomposed as:
            z = ReLU( W_A · x  +  b_A )     dense + ReLU
            y = w_B · z                       dense (linear)

        Works element-wise on arrays of any shape.
        """
        shape = x.shape
        x_col = x.reshape(-1, 1)                                         # (m, 1)
        z = relu(x_col @ self.W_mod_A.T + self.b_mod_A[np.newaxis, :])  # (m, V+1)
        y = z @ self.w_mod_B                                             # (m,)
        return np.round(y).reshape(shape)

    # ----------------------------------------------------------------
    #  Forward pass
    # ----------------------------------------------------------------

    def forward(self, input_image, verbose=True):
        """Run the full 6-layer CNN pipeline.

        Args:
            input_image: n x (2*n_bits) binary image.
            verbose:     Print layer dimensions to stdout.
        Returns:
            output:  (2n-1) x output_bits  binary (0/1) image.
            layers:  List of (name, image_array) for visualisation.
        """
        n, nb = self.n, self.n_bits
        layers = [("Input", input_image.copy())]

        # ---- Layer 1: Split (Reshape) --------------------------------
        P_bits = input_image[:, :nb]          # n x 20
        Q_bits = input_image[:, nb:]          # n x 20
        if verbose:
            print(f"  Layer 1 (Split):     {input_image.shape} "
                  f"-> P({n}x{nb}) + Q({n}x{nb})")
        layers.append(("Layer 1: P bits", P_bits.copy()))
        layers.append(("Layer 1: Q bits", Q_bits.copy()))

        # ---- Layer 2: Broadcast / Tile --------------------------------
        #  For each of 20x20 = 400 bit pairs (p, q) create two n x n
        #  matrices by broadcasting P[:,p] across columns and Q[:,q]
        #  across rows.  This is the CNN "upsample / tile" operation.
        P_tiled = np.zeros((self.n_pairs, n, n), dtype=np.float64)
        Q_tiled = np.zeros((self.n_pairs, n, n), dtype=np.float64)
        for p in range(nb):
            for q in range(nb):
                idx = p * nb + q
                P_tiled[idx] = broadcast_to_matrix(P_bits[:, p], n, axis=0)
                Q_tiled[idx] = broadcast_to_matrix(Q_bits[:, q], n, axis=1)
        if verbose:
            print(f"  Layer 2 (Broadcast): {self.n_pairs} pairs "
                  f"of ({n}x{n}) matrices")
        layers.append(("Layer 2: P tiled (p=0)", P_tiled[0].copy()))
        layers.append(("Layer 2: Q tiled (q=0)", Q_tiled[0].copy()))

        # ---- Layer 3: Binary AND via ReLU -----------------------------
        #  M_{p,q}[i,j] = ReLU(P_tile + Q_tile - 1)
        #  For binary inputs this equals a_{i,p} AND b_{j,q}.
        M = relu(P_tiled + Q_tiled - 1.0)
        if verbose:
            print(f"  Layer 3 (AND/ReLU):  {self.n_pairs} binary "
                  f"({n}x{n}) matrices")
        layers.append(("Layer 3: AND (p=0,q=0)", M[0].copy()))
        layers.append(("Layer 3: All AND channels",
                        self._channels_to_grid(M, nb)))

        # ---- Layer 4: Anti-diagonal Summation (Dense) -----------------
        #  S_{p,q}[k] = sum_{i+j=k} M_{p,q}[i,j]
        #  Implemented as: S = flatten(M) @ W_diag^T
        M_flat = M.reshape(self.n_pairs, n * n)           # (400, n^2)
        S = np.round(dense(M_flat, self.W_diag)).T        # (2n-1, 400)
        if verbose:
            print(f"  Layer 4 (Diag Pool): ({self.out_len}x{self.n_pairs})")
        layers.append(("Layer 4: Diagonal sums", S.copy()))

        # ---- Layer 5: Bit-weight Grouping (1x1 Conv / Channel Mix) ----
        #  T[k, r] = sum_{p+q=r} S[k, p*nb+q]
        T = np.round(dense(S, self.W_group))              # (2n-1, 39)
        if verbose:
            print(f"  Layer 5 (Grouping):  ({self.out_len}x{self.max_r + 1})")
        layers.append(("Layer 5: Bit-weight groups", T.copy()))

        # ---- Layer 6: Carry Propagation (Dense + ReLU loop) -----------
        #  Process each bit position r = 0 … output_bits-1:
        #    bit_r  = T_ext[:, r] mod 2        (Dense + ReLU + Dense)
        #    carry  = (T_ext[:, r] - bit_r) / 2  (linear)
        #    T_ext[:, r+1] += carry
        T_ext = np.zeros((self.out_len, self.output_bits), dtype=np.float64)
        T_ext[:, :self.max_r + 1] = T

        output = np.zeros((self.out_len, self.output_bits), dtype=np.float64)
        for r in range(self.output_bits):
            col = T_ext[:, r]
            bit_r = self._mod2_relu(col)
            output[:, r] = bit_r
            carry = np.round((col - bit_r) * 0.5)
            if r + 1 < self.output_bits:
                T_ext[:, r + 1] += carry

        if verbose:
            print(f"  Layer 6 (Carry):     ({self.out_len}x{self.output_bits}) "
                  f"binary output")
        layers.append(("Layer 6: After carry", T_ext.copy()))
        layers.append(("Output", output.copy()))

        return output, layers

    # ----------------------------------------------------------------
    #  Helpers
    # ----------------------------------------------------------------

    def _channels_to_grid(self, M, nb):
        """Tile all channel matrices into an (nb*n) x (nb*n) image grid."""
        n = self.n
        grid = np.zeros((nb * n, nb * n), dtype=M.dtype)
        for p in range(nb):
            for q in range(nb):
                grid[p * n:(p + 1) * n,
                     q * n:(q + 1) * n] = M[p * nb + q]
        return grid


# ================================================================
#  Part 4 — Visualisation
# ================================================================

BW_CMAP = ListedColormap(["white", "black"])


def visualize_pipeline(layer_outputs, n_bits=20, save_path=None):
    """One-row overview of key layers in the pipeline."""
    show = [
        "Input",
        "Layer 1: P bits",
        "Layer 1: Q bits",
        "Layer 3: All AND channels",
        "Layer 4: Diagonal sums",
        "Layer 5: Bit-weight groups",
        "Output",
    ]
    selected = [(nm, img) for nm, img in layer_outputs if nm in show]
    N = len(selected)

    fig, axes = plt.subplots(1, N, figsize=(3.5 * N, 4))
    if N == 1:
        axes = [axes]

    for ax, (name, img) in zip(axes, selected):
        is_binary = np.all((img == 0) | (img == 1))
        if is_binary:
            ax.imshow(img, cmap=BW_CMAP, vmin=0, vmax=1,
                      aspect="auto", interpolation="nearest")
        else:
            im = ax.imshow(img, cmap="viridis", aspect="auto",
                           interpolation="nearest")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(name, fontsize=8, fontweight="bold")

    plt.suptitle("CNN Polynomial Convolution — Layer Overview",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def visualize_detail(layer_outputs, n_bits=20, save_path=None):
    """Detailed 2 x 3 subplot view of selected layers."""
    d = {nm: img for nm, img in layer_outputs}

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (0,0) Input image with P|Q boundary
    ax = axes[0, 0]
    ax.imshow(d["Input"], cmap=BW_CMAP, vmin=0, vmax=1,
              aspect="auto", interpolation="nearest")
    ax.axvline(x=n_bits - 0.5, color="red", lw=2, ls="--",
               label="P | Q boundary")
    ax.set_title("Input: n x 40 binary")
    ax.set_xlabel("Bit position  (P left | Q right)")
    ax.set_ylabel("Coefficient index")
    ax.legend(fontsize=7)

    # (0,1) Example AND channel (p=0, q=0)
    ax = axes[0, 1]
    key = "Layer 3: AND (p=0,q=0)"
    if key in d:
        ax.imshow(d[key], cmap=BW_CMAP, vmin=0, vmax=1,
                  aspect="auto", interpolation="nearest")
    ax.set_title("Layer 3: AND channel (p=0, q=0)")
    ax.set_xlabel("j  (Q coefficient index)")
    ax.set_ylabel("i  (P coefficient index)")

    # (0,2) Diagonal sums
    ax = axes[0, 2]
    im = ax.imshow(d["Layer 4: Diagonal sums"], cmap="viridis",
                   aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Layer 4: Anti-diagonal sums  S[k, (p,q)]")
    ax.set_xlabel("Channel index  (p*20 + q)")
    ax.set_ylabel("Output coefficient k")

    # (1,0) Bit-weight groups
    ax = axes[1, 0]
    im = ax.imshow(d["Layer 5: Bit-weight groups"], cmap="viridis",
                   aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Layer 5: Bit-weight groups  T[k, r]")
    ax.set_xlabel("Bit weight  r = p + q")
    ax.set_ylabel("Output coefficient k")

    # (1,1) State after carry propagation
    ax = axes[1, 1]
    im = ax.imshow(d["Layer 6: After carry"], cmap="plasma",
                   aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Layer 6: State after carry propagation")
    ax.set_xlabel("Bit position")
    ax.set_ylabel("Output coefficient k")

    # (1,2) Final output
    ax = axes[1, 2]
    ax.imshow(d["Output"], cmap=BW_CMAP, vmin=0, vmax=1,
              aspect="auto", interpolation="nearest")
    ax.set_title("Output: binary encoding of product")
    ax.set_xlabel("Bit position")
    ax.set_ylabel("Coefficient index")

    plt.suptitle("Convoluted — CNN Polynomial Convolution (Detail)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ================================================================
#  Part 5 — Main Demo
# ================================================================

def main():
    print("=" * 62)
    print("  Convoluted — CNN-Based Polynomial Convolution")
    print("=" * 62)

    # ----- parameters -----
    n = 4
    n_bits = 20

    P = [3, 5, 2, 1]        # P(x) = 3 + 5x + 2x^2 + x^3
    Q = [1, 4, 3, 2]        # Q(x) = 1 + 4x + 3x^2 + 2x^3
    expected = np.convolve(P, Q)

    print(f"\n  P(x) = {P[0]} + {P[1]}x + {P[2]}x^2 + {P[3]}x^3")
    print(f"  Q(x) = {Q[0]} + {Q[1]}x + {Q[2]}x^2 + {Q[3]}x^3")
    print(f"  Expected product coefficients: {expected.tolist()}")

    # ----- encode -----
    input_image = encode_input(P, Q, n_bits)
    print(f"\n  Input image shape: {input_image.shape}  (binary 0/1)")

    # ----- build CNN -----
    cnn = PolynomialConvCNN(n=n, n_bits=n_bits)
    print(f"  CNN constructed:  n={n}, n_bits={n_bits}")
    print(f"    Output coefficients : {cnn.out_len}")
    print(f"    Output bits / coeff : {cnn.output_bits}")
    print(f"    Max carry value     : {cnn.global_v_max}")
    print(f"    Mod-2 ReLU units    : {cnn.global_v_max + 1}")

    # ----- forward pass -----
    print(f"\n  Forward pass through 6 layers:")
    output, layers = cnn.forward(input_image)

    # ----- decode & verify -----
    result = decode_polynomial(output)
    print(f"\n  Decoded result : {result.tolist()}")
    print(f"  NumPy convolve : {expected.tolist()}")
    match = np.array_equal(result, expected)
    print(f"  Verification   : {'PASS' if match else '*** FAIL ***'}")

    # ----- additional tests -----
    print(f"\n  Additional tests (n={n}, n_bits={n_bits}):")
    tests = [
        ([1, 0, 0, 0], [1, 0, 0, 0], "1 * 1"),
        ([0, 1, 0, 0], [0, 1, 0, 0], "x * x"),
        ([1, 1, 1, 1], [1, 1, 1, 1], "all-ones * all-ones"),
        ([1048575] * 4, [1, 0, 0, 0], "max_coeff * unit"),
        ([1048575] * 4, [1048575] * 4, "max_coeff * max_coeff"),
        ([100, 200, 300, 400], [10, 20, 30, 40], "medium values"),
        ([999999, 500000, 123456, 789012],
         [2, 3, 5, 7], "large * small"),
    ]
    all_pass = True
    for Pt, Qt, name in tests:
        inp = encode_input(Pt, Qt, n_bits)
        out, _ = cnn.forward(inp, verbose=False)
        res = decode_polynomial(out)
        exp = np.convolve(Pt, Qt)
        ok = np.array_equal(res, exp)
        status = "OK" if ok else "FAIL"
        print(f"    [{status}]  {name}")
        if not ok:
            print(f"           got:      {res.tolist()}")
            print(f"           expected: {exp.tolist()}")
            all_pass = False

    print(f"\n  All tests {'passed!' if all_pass else 'FAILED!'}")

    # ----- visualise -----
    print("\n  Generating visualisations...")
    visualize_pipeline(layers, n_bits, save_path="layer_overview.png")
    visualize_detail(layers, n_bits, save_path="layer_detail.png")


if __name__ == "__main__":
    main()
