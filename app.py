"""
Interactive UI for Convoluted CNN polynomial multiplication.

Run with:
    streamlit run app.py
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import ListedColormap

from convoluted import PolynomialConvCNN, decode_polynomial, encode_input


BW_CMAP = ListedColormap(["#f7f9fc", "#1a1d24"])


def format_polynomial(coeffs, var="x"):
    """Create a readable polynomial string from coefficient list."""
    terms = []
    for i, c in enumerate(coeffs):
        if c == 0:
            continue
        if i == 0:
            terms.append(str(c))
        elif i == 1:
            terms.append(f"{c}{var}")
        else:
            terms.append(f"{c}{var}^{i}")
    return " + ".join(terms) if terms else "0"


def render_coefficient_inputs(label, n, max_value, defaults, key_prefix):
    """Render compact coefficient controls and return values."""
    st.markdown(f"**{label} coefficients**")
    values = []
    cols_per_row = 4
    for start in range(0, n, cols_per_row):
        cols = st.columns(cols_per_row)
        for offset, col in enumerate(cols):
            idx = start + offset
            if idx >= n:
                continue
            default_value = defaults[idx] if idx < len(defaults) else 0
            with col:
                val = st.number_input(
                    f"{label}[{idx}]",
                    min_value=0,
                    max_value=max_value,
                    value=int(default_value),
                    step=1,
                    key=f"{key_prefix}_{n}_{idx}",
                )
            values.append(int(val))
    return values


def show_matrix(matrix, title):
    """Display matrix as heatmap and table."""
    is_binary = bool(np.all((matrix == 0) | (matrix == 1)))
    rows, cols = matrix.shape
    fig_w = max(6, min(14, cols * 0.28))
    fig_h = max(3, min(8, rows * 0.32))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    if is_binary:
        im = ax.imshow(
            matrix,
            cmap=BW_CMAP,
            vmin=0,
            vmax=1,
            aspect="auto",
            interpolation="nearest",
        )
    else:
        im = ax.imshow(
            matrix,
            cmap="magma",
            aspect="auto",
            interpolation="nearest",
        )
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)

    ax.set_title(title)
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.dataframe(matrix.astype(np.int64), use_container_width=True)


def run():
    st.set_page_config(page_title="Convoluted UI", page_icon="##", layout="wide")
    st.markdown(
        """
        <style>
          .block-container {padding-top: 1.4rem; padding-bottom: 1.4rem;}
          .stButton > button {
              width: 100%;
              border-radius: 10px;
              border: 1px solid #2f3a4f;
              background: linear-gradient(180deg, #2f80ed 0%, #2167bf 100%);
              color: white;
              font-weight: 600;
          }
          [data-testid="stMetricValue"] {font-size: 1.45rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Convolution Squared")
    st.caption(
        "Enter polynomial degree and coefficients, then inspect every layer matrix."
    )

    left, right = st.columns([1.0, 1.25], gap="large")

    with left:
        st.subheader("Inputs")
        degree = st.number_input("Degree", min_value=0, max_value=12, value=3, step=1)
        n = int(degree) + 1

        with st.expander("Advanced settings", expanded=False):
            n_bits = st.number_input(
                "Bits per coefficient",
                min_value=4,
                max_value=24,
                value=20,
                step=1,
            )
            st.caption("Each coefficient must be in [0, 2^bits - 1].")

        max_coeff = (1 << int(n_bits)) - 1
        default_p = [3, 5, 2, 1] + [0] * max(0, n - 4)
        default_q = [1, 4, 3, 2] + [0] * max(0, n - 4)

        P = render_coefficient_inputs("P", n, max_coeff, default_p, "P")
        Q = render_coefficient_inputs("Q", n, max_coeff, default_q, "Q")

        st.markdown("")
        compute_clicked = st.button("Compute Layer Matrices", type="primary")

        if compute_clicked:
            st.session_state["compute"] = {
                "degree": int(degree),
                "n_bits": int(n_bits),
                "P": P,
                "Q": Q,
            }

    with right:
        if "compute" not in st.session_state:
            st.info("Set inputs and click **Compute Layer Matrices**.")
            return

        params = st.session_state["compute"]
        n = params["degree"] + 1
        n_bits = params["n_bits"]
        P = params["P"]
        Q = params["Q"]

        input_image = encode_input(P, Q, n_bits=n_bits)
        cnn = PolynomialConvCNN(n=n, n_bits=n_bits)
        output, layers = cnn.forward(input_image, verbose=False)
        result = decode_polynomial(output)
        expected = np.convolve(P, Q)
        verified = np.array_equal(result, expected)

        st.subheader("Result")
        c1, c2, c3 = st.columns(3)
        c1.metric("Output coefficients", int(cnn.out_len))
        c2.metric("Output bits / coeff", int(cnn.output_bits))
        c3.metric("Verification", "PASS" if verified else "FAIL")

        st.markdown(f"`P(x) = {format_polynomial(P)}`")
        st.markdown(f"`Q(x) = {format_polynomial(Q)}`")
        st.markdown(f"`R(x) = {format_polynomial(result.tolist())}`")
        st.write("Result coefficients:", result.tolist())

        tabs = st.tabs(["Layer outputs", "Weight matrices"])

        with tabs[0]:
            layer_names = [name for name, _ in layers]
            selected = st.selectbox("Layer to inspect", options=layer_names)
            selected_matrix = dict(layers)[selected]
            st.caption(f"Shape: {selected_matrix.shape[0]} x {selected_matrix.shape[1]}")
            show_matrix(selected_matrix, selected)

        with tabs[1]:
            weight_options = {
                "W_diag (Layer 4 anti-diagonal)": cnn.W_diag,
                "W_group (Layer 5 bit-weight grouping)": cnn.W_group,
                "W_mod_A (Layer 6 dense A)": cnn.W_mod_A,
                "w_mod_B (Layer 6 dense B weights)": cnn.w_mod_B.reshape(1, -1),
            }
            selected_w = st.selectbox(
                "Weight matrix to inspect",
                options=list(weight_options.keys()),
            )
            W = weight_options[selected_w]
            st.caption(f"Shape: {W.shape[0]} x {W.shape[1]}")
            show_matrix(W, selected_w)


if __name__ == "__main__":
    run()
