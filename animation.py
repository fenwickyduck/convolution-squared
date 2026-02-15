"""
Convoluted — Manim Animation
=============================
Animated explanation of the CNN polynomial convolution algorithm.

Render commands:
    python -m manim -ql  animation.py ConvolutedAnimation   (low quality, fast)
    python -m manim -qm  animation.py ConvolutedAnimation   (medium quality)
    python -m manim -qh  animation.py ConvolutedAnimation   (high quality)
"""

from manim import *

# ══════════════════════════════════════════════════════════════
#  Hardcoded data from the running example
# ══════════════════════════════════════════════════════════════

P_COEFFS = [3, 5, 2, 1]
Q_COEFFS = [1, 4, 3, 2]
PRODUCT = [3, 17, 31, 30, 20, 7, 2]

# Binary encodings (LSB first, 3 significant bits)
P_BITS = [[1, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 0]]
Q_BITS = [[1, 0, 0], [0, 0, 1], [1, 1, 0], [0, 1, 0]]
INPUT_IMG = [P_BITS[i] + Q_BITS[i] for i in range(4)]

# Layer 2-3 example for bit-pair (p=0, q=0)
P_COL0 = [1, 1, 0, 1]
Q_COL0 = [1, 0, 1, 0]
P_TILED = [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]]
Q_TILED = [[1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]]
SUM_M1 = [[1, 0, 1, 0], [1, 0, 1, 0], [0, -1, 0, -1], [1, 0, 1, 0]]
AND_MAT = [[1, 0, 1, 0], [1, 0, 1, 0], [0, 0, 0, 0], [1, 0, 1, 0]]
ADIAG = [1, 1, 1, 2, 0, 1, 0]

# Layer 5 grouping example — for k=3 (c_3 = 30)
# 9 bit-pair channels grouped by r = p + q
GROUPS_BY_R = {
    0: [(0, 0)],
    1: [(0, 1), (1, 0)],
    2: [(0, 2), (1, 1), (2, 0)],
    3: [(1, 2), (2, 1)],
    4: [(2, 2)],
}
# S_{p,q}[3] values for each channel (at k=3)
S_K3 = {
    (0, 0): 2, (0, 1): 2, (0, 2): 0,
    (1, 0): 0, (1, 1): 1, (1, 2): 1,
    (2, 0): 1, (2, 1): 1, (2, 2): 0,
}
# T[3, r] = sum of S_{p,q}[3] for p+q=r
T_K3 = [2, 2, 2, 2, 0]

# Layer 6 carry propagation for k=3: T = [2, 2, 2, 2, 0]
#   r=0: T=2 → bit=0, carry=1 → T[1] becomes 3
#   r=1: T=3 → bit=1, carry=1 → T[2] becomes 3
#   r=2: T=3 → bit=1, carry=1 → T[3] becomes 3
#   r=3: T=3 → bit=1, carry=1 → T[4] becomes 1
#   r=4: T=1 → bit=1, carry=0 → done
# Output bits: [0, 1, 1, 1, 1] → 30
CARRY_STEPS = [
    # (r, T_before, bit, carry, T_next_after)
    (0, 2, 0, 1, 3),
    (1, 3, 1, 1, 3),
    (2, 3, 1, 1, 3),
    (3, 3, 1, 1, 1),
    (4, 1, 1, 0, None),
]
CARRY_RESULT_BITS = [0, 1, 1, 1, 1]

# Output bits (LSB first, 5 bits shown)
OUT_BITS = [
    [1, 1, 0, 0, 0],   # 3
    [1, 0, 0, 0, 1],   # 17
    [1, 1, 1, 1, 1],   # 31
    [0, 1, 1, 1, 1],   # 30
    [0, 0, 1, 0, 1],   # 20
    [1, 1, 1, 0, 0],   # 7
    [0, 1, 0, 0, 0],   # 2
]

# ══════════════════════════════════════════════════════════════
#  Colors
# ══════════════════════════════════════════════════════════════

C_ONE = "#3a86ff"
C_ZERO = "#d4dbe8"
C_NEG = "#ef476f"
C_OK = "#06d6a0"
C_CARRY = "#ff9f1c"

# Unicode helpers
SUB = str.maketrans("0123456789", "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089")

# Group colors for r values
R_COLORS = [RED_B, ORANGE, GOLD_B, GREEN_B, TEAL_B]

# ══════════════════════════════════════════════════════════════
#  Grid builder
# ══════════════════════════════════════════════════════════════


def build_grid(data, cs=0.5, show_text=True):
    """Create a centered grid of colored cells.

    Returns (VGroup containing all cells,
             cells[i][j] — each a VGroup(square, text)).
    """
    rows, cols = len(data), len(data[0])
    grp = VGroup()
    cells = []
    for i in range(rows):
        row = []
        for j in range(cols):
            v = data[i][j]
            sq = Square(side_length=cs, stroke_color=GRAY_B, stroke_width=1.2)
            if v == 1:
                sq.set_fill(C_ONE, opacity=0.9)
            elif v == 0:
                sq.set_fill(C_ZERO, opacity=0.85)
            elif v < 0:
                sq.set_fill(C_NEG, opacity=0.9)
            else:
                sq.set_fill(C_ONE, opacity=min(0.5 + v * 0.1, 0.95))

            x = (j - (cols - 1) / 2) * cs
            y = ((rows - 1) / 2 - i) * cs
            sq.move_to([x, y, 0])

            if show_text:
                fs = max(12, int(cs * 30))
                t = Text(str(int(v)), font_size=fs)
                t.set_color(WHITE if v != 0 else GRAY_C)
                t.move_to(sq)
                cell = VGroup(sq, t)
            else:
                cell = VGroup(sq)
            grp.add(cell)
            row.append(cell)
        cells.append(row)
    return grp, cells


def build_column(values, cs=0.5):
    """Build a vertical strip of cells (Nx1 grid)."""
    return build_grid([[v] for v in values], cs)


def value_cell(val, cs=0.6, color=None):
    """Build a single square with a number inside."""
    sq = Square(side_length=cs, stroke_color=GRAY_B, stroke_width=1.5)
    if color is not None:
        sq.set_fill(color, opacity=0.85)
    elif val == 0:
        sq.set_fill(C_ZERO, opacity=0.85)
    elif val == 1:
        sq.set_fill(C_ONE, opacity=0.9)
    else:
        sq.set_fill(C_ONE, opacity=min(0.5 + val * 0.1, 0.95))
    t = Text(str(int(val)), font_size=max(14, int(cs * 28)))
    t.set_color(WHITE if val != 0 else GRAY_C)
    t.move_to(sq)
    return VGroup(sq, t)


# ══════════════════════════════════════════════════════════════
#  Main scene
# ══════════════════════════════════════════════════════════════


class ConvolutedAnimation(Scene):
    def construct(self):
        self.camera.background_color = "#0f0f1a"

        self.scene_title()
        self.scene_encoding()
        self.scene_relu_and()
        self.scene_layer1()
        self.scene_layer2()
        self.scene_layer3()
        self.scene_layer4()
        self.scene_layer5()
        self.scene_layer6()
        self.scene_result()

    # ── utilities ────────────────────────────────────────────

    def wipe(self):
        if self.mobjects:
            self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.4)

    def header(self, text, color=BLUE_B):
        h = Text(text, font_size=28, color=color, weight="BOLD")
        h.to_edge(UP, buff=0.35)
        self.play(FadeIn(h, shift=DOWN * 0.15), run_time=0.4)
        return h

    # ─────────────────────────────────────────────────────────
    # 1  TITLE
    # ─────────────────────────────────────────────────────────

    def scene_title(self):
        title = Text(
            "Polynomial Multiplication\nvia CNN",
            font_size=48, weight="BOLD", line_spacing=1.3,
        )
        title.set_color_by_gradient(BLUE_B, TEAL_B)
        self.play(Write(title), run_time=1.5)
        self.wait(1)

        sub = Text(
            "Using only standard CNN layers on binary images",
            font_size=20, color=GRAY_B,
        )
        sub.next_to(title, DOWN, buff=0.6)
        self.play(FadeIn(sub), run_time=0.6)
        self.wait(0.5)

        p_eq = Text("P(x) = 3 + 5x + 2x\u00b2 + x\u00b3",
                     font_size=28, color=BLUE_B)
        q_eq = Text("Q(x) = 1 + 4x + 3x\u00b2 + 2x\u00b3",
                     font_size=28, color=TEAL_B)
        goal = Text("R(x) = P(x) \u00b7 Q(x) = ?",
                     font_size=32, color=YELLOW)
        eqs = VGroup(p_eq, q_eq, goal).arrange(DOWN, buff=0.25)
        eqs.next_to(sub, DOWN, buff=0.5)

        self.play(FadeIn(p_eq), run_time=0.5)
        self.play(FadeIn(q_eq), run_time=0.5)
        self.play(Write(goal), run_time=0.7)
        self.wait(2)
        self.wipe()

    # ─────────────────────────────────────────────────────────
    # 2  BINARY ENCODING
    # ─────────────────────────────────────────────────────────

    def scene_encoding(self):
        self.header("Encode Coefficients as Binary")

        p_title = Text("P = [3, 5, 2, 1]", font_size=22, color=BLUE_B)
        p_title.move_to(LEFT * 3 + UP * 1.8)
        pg, _ = build_grid(P_BITS, cs=0.55)
        pg.next_to(p_title, DOWN, buff=0.35)

        q_title = Text("Q = [1, 4, 3, 2]", font_size=22, color=TEAL_B)
        q_title.move_to(RIGHT * 3 + UP * 1.8)
        qg, _ = build_grid(Q_BITS, cs=0.55)
        qg.next_to(q_title, DOWN, buff=0.35)

        self.play(FadeIn(p_title), FadeIn(pg), run_time=0.8)
        self.play(FadeIn(q_title), FadeIn(qg), run_time=0.8)

        explain = Text(
            "Each row = one coefficient in binary (bit 0 = LSB at left)",
            font_size=17, color=GRAY_B,
        )
        explain.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(explain), run_time=0.4)
        self.wait(1.5)

        # Combine into single input image
        self.play(
            FadeOut(p_title), FadeOut(q_title),
            FadeOut(pg), FadeOut(qg), FadeOut(explain),
            run_time=0.4,
        )

        cg, _ = build_grid(INPUT_IMG, cs=0.55)
        cg.move_to(DOWN * 0.2)
        sep_x = cg.get_center()[0]
        sep = DashedLine(
            [sep_x, cg.get_top()[1] + 0.15, 0],
            [sep_x, cg.get_bottom()[1] - 0.15, 0],
            color=RED, stroke_width=2, dash_length=0.08,
        )
        p_lab = Text("P", font_size=20, color=BLUE_B)
        q_lab = Text("Q", font_size=20, color=TEAL_B)
        p_lab.next_to(cg, UP, buff=0.2).shift(LEFT * 0.85)
        q_lab.next_to(cg, UP, buff=0.2).shift(RIGHT * 0.85)
        inp_label = Text(
            "Input image: 4 \u00d7 6 binary pixels",
            font_size=22, color=WHITE,
        )
        inp_label.next_to(cg, DOWN, buff=0.3)

        self.play(
            FadeIn(cg), Create(sep),
            FadeIn(p_lab), FadeIn(q_lab), FadeIn(inp_label),
            run_time=0.8,
        )
        self.wait(1.5)
        self.wipe()

    # ─────────────────────────────────────────────────────────
    # 3  AND = ReLU TRICK
    # ─────────────────────────────────────────────────────────

    def scene_relu_and(self):
        self.header("Key Insight: AND via ReLU")

        eq = Text(
            "a \u00b7 b  =  ReLU(a + b \u2212 1)",
            font_size=36, color=YELLOW,
        )
        eq.shift(UP * 1.2)
        note = Text(
            "for binary a, b \u2208 {0, 1}", font_size=18, color=GRAY_B,
        )
        note.next_to(eq, DOWN, buff=0.2)

        self.play(Write(eq), run_time=1.0)
        self.play(FadeIn(note), run_time=0.4)
        self.wait(0.5)

        # Truth table
        hdrs = ["a", "b", "a+b\u22121", "ReLU", "a\u00b7b"]
        rows = [
            ["0", "0", "\u22121", "0", "0"],
            ["0", "1", " 0", "0", "0"],
            ["1", "0", " 0", "0", "0"],
            ["1", "1", " 1", "1", "1"],
        ]

        table = VGroup()
        for j, h in enumerate(hdrs):
            t = Text(h, font_size=17, color=BLUE_B, weight="BOLD")
            t.move_to([j * 1.2 - 2.4, -0.2, 0])
            table.add(t)
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                t = Text(val, font_size=17)
                if i == 3 and j >= 3:
                    t.set_color(YELLOW)
                elif val.strip().startswith("\u2212"):
                    t.set_color(C_NEG)
                else:
                    t.set_color(WHITE)
                t.move_to([j * 1.2 - 2.4, -0.65 - i * 0.35, 0])
                table.add(t)

        self.play(FadeIn(table), run_time=0.7)

        takeaway = Text(
            "Bit multiplication = addition + bias + ReLU",
            font_size=20, color=C_OK, weight="BOLD",
        )
        takeaway.to_edge(DOWN, buff=0.6)
        self.play(FadeIn(takeaway), run_time=0.5)
        self.wait(2)
        self.wipe()

    # ─────────────────────────────────────────────────────────
    # 4  LAYER 1 — SPLIT
    # ─────────────────────────────────────────────────────────

    def scene_layer1(self):
        self.header("Layer 1: Split (Reshape)")

        cg, _ = build_grid(INPUT_IMG, cs=0.55)
        cg.shift(UP * 0.3)
        lab = Text("Input 4 \u00d7 6", font_size=18, color=GRAY_B)
        lab.next_to(cg, UP, buff=0.15)
        self.play(FadeIn(cg), FadeIn(lab), run_time=0.5)
        self.wait(0.5)

        pg, _ = build_grid(P_BITS, cs=0.55)
        pg.shift(LEFT * 2.5 + DOWN * 0.5)
        pl = Text("P bits (4\u00d73)", font_size=16, color=BLUE_B)
        pl.next_to(pg, DOWN, buff=0.1)

        qg, _ = build_grid(Q_BITS, cs=0.55)
        qg.shift(RIGHT * 2.5 + DOWN * 0.5)
        ql = Text("Q bits (4\u00d73)", font_size=16, color=TEAL_B)
        ql.next_to(qg, DOWN, buff=0.1)

        self.play(
            FadeOut(cg), FadeOut(lab),
            FadeIn(pg), FadeIn(qg),
            FadeIn(pl), FadeIn(ql),
            run_time=0.8,
        )
        self.wait(1.5)
        self.wipe()

    # ─────────────────────────────────────────────────────────
    # 5  LAYER 2 — BROADCAST
    # ─────────────────────────────────────────────────────────

    def scene_layer2(self):
        self.header("Layer 2: Broadcast (Tile)")

        # P column 0 -> tiled P
        pcol, _ = build_column(P_COL0, cs=0.5)
        pcol.shift(LEFT * 5 + DOWN * 0.2)
        pcol_lab = Text("P[:,0]", font_size=18, color=BLUE_B)
        pcol_lab.next_to(pcol, UP, buff=0.12)

        self.play(FadeIn(pcol), FadeIn(pcol_lab), run_time=0.5)

        arr_p = Arrow(LEFT * 3.9, LEFT * 3.1, color=WHITE, stroke_width=2,
                       buff=0.05, max_tip_length_to_length_ratio=0.3)
        arr_p.shift(DOWN * 0.2)

        ptg, _ = build_grid(P_TILED, cs=0.45)
        ptg.shift(LEFT * 1.9 + DOWN * 0.2)
        ptg_lab = Text("P tiled (4\u00d74)", font_size=13, color=BLUE_B)
        ptg_lab.next_to(ptg, DOWN, buff=0.1)

        self.play(
            GrowArrow(arr_p), FadeIn(ptg), FadeIn(ptg_lab),
            run_time=0.7,
        )
        self.wait(0.3)

        # Q column 0 -> tiled Q
        qcol, _ = build_column(Q_COL0, cs=0.5)
        qcol.shift(RIGHT * 0.5 + DOWN * 0.2)
        qcol_lab = Text("Q[:,0]", font_size=18, color=TEAL_B)
        qcol_lab.next_to(qcol, UP, buff=0.12)

        arr_q = Arrow(RIGHT * 1.6, RIGHT * 2.4, color=WHITE, stroke_width=2,
                       buff=0.05, max_tip_length_to_length_ratio=0.3)
        arr_q.shift(DOWN * 0.2)

        qtg, _ = build_grid(Q_TILED, cs=0.45)
        qtg.shift(RIGHT * 3.6 + DOWN * 0.2)
        qtg_lab = Text("Q tiled (4\u00d74)", font_size=13, color=TEAL_B)
        qtg_lab.next_to(qtg, DOWN, buff=0.1)

        self.play(
            FadeIn(qcol), FadeIn(qcol_lab),
            GrowArrow(arr_q), FadeIn(qtg), FadeIn(qtg_lab),
            run_time=0.7,
        )

        note = Text(
            "Repeat for all 3 \u00d7 3 = 9 bit-position pairs \u2192 9 channels",
            font_size=15, color=GRAY_B,
        )
        note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(note), run_time=0.4)
        self.wait(1.5)
        self.wipe()

    # ─────────────────────────────────────────────────────────
    # 6  LAYER 3 — AND VIA ReLU
    # ─────────────────────────────────────────────────────────

    def scene_layer3(self):
        self.header("Layer 3: AND via ReLU")

        sg, _ = build_grid(SUM_M1, cs=0.6)
        sg.shift(LEFT * 3 + DOWN * 0.1)
        sl = Text("P_tile + Q_tile \u2212 1", font_size=20, color=WHITE)
        sl.next_to(sg, UP, buff=0.15)

        self.play(FadeIn(sg), FadeIn(sl), run_time=0.6)
        self.wait(0.8)

        relu_arr = Arrow(LEFT * 1.1, RIGHT * 0.6, color=YELLOW,
                          stroke_width=3, buff=0.1)
        relu_arr.shift(DOWN * 0.1)
        relu_lab = Text("ReLU", font_size=20, color=YELLOW, weight="BOLD")
        relu_lab.next_to(relu_arr, UP, buff=0.08)

        ag, _ = build_grid(AND_MAT, cs=0.6)
        ag.shift(RIGHT * 3 + DOWN * 0.1)
        al = Text("M\u2080,\u2080", font_size=22, color=WHITE)
        al.next_to(ag, UP, buff=0.15)

        self.play(
            GrowArrow(relu_arr), FadeIn(relu_lab),
            FadeIn(ag), FadeIn(al),
            run_time=0.8,
        )

        note = Text(
            "Negatives \u2192 0    Positives stay    =    exact binary AND",
            font_size=17, color=GRAY_B,
        )
        note.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(note), run_time=0.4)
        self.wait(1.5)
        self.wipe()

    # ─────────────────────────────────────────────────────────
    # 7  LAYER 4 — ANTI-DIAGONAL SUMMATION
    # ─────────────────────────────────────────────────────────

    def scene_layer4(self):
        self.header("Layer 4: Anti-Diagonal Pooling")

        ag, cells = build_grid(AND_MAT, cs=0.65)
        ag.shift(LEFT * 2 + DOWN * 0.2)

        # Row / column labels
        for i in range(4):
            il = Text(f"i={i}", font_size=11, color=GRAY_B)
            il.next_to(cells[i][0], LEFT, buff=0.12)
            ag.add(il)
        for j in range(4):
            jl = Text(f"j={j}", font_size=11, color=GRAY_B)
            jl.next_to(cells[0][j], UP, buff=0.08)
            ag.add(jl)

        self.play(FadeIn(ag), run_time=0.5)

        formula = Text(
            "S[k] = \u03a3 M[i,j]   where i+j = k",
            font_size=20, color=YELLOW,
        )
        formula.shift(RIGHT * 3 + UP * 2.3)
        self.play(Write(formula), run_time=0.6)

        # Build result column (filled incrementally)
        diag_colors = [RED_B, ORANGE, GOLD_B, GREEN_B, TEAL_B, BLUE_B, PURPLE_B]
        res_cells = VGroup()
        res_labels = VGroup()
        col_x = 3.5
        for k in range(7):
            v = ADIAG[k]
            sq = Square(side_length=0.5, stroke_color=GRAY_B, stroke_width=1.2)
            sq.set_fill(C_ONE if v > 0 else C_ZERO, opacity=0.85)
            sq.move_to([col_x, (3 - k) * 0.5, 0])
            t = Text(str(v), font_size=15, color=WHITE if v > 0 else GRAY_C)
            t.move_to(sq)
            res_cells.add(VGroup(sq, t))
            kl = Text(f"k={k}", font_size=10, color=GRAY_B)
            kl.next_to(sq, RIGHT, buff=0.08)
            res_labels.add(kl)

        s_lab = Text("S[k]", font_size=16, color=WHITE)
        s_lab.move_to([col_x, 2.3, 0])
        self.play(FadeIn(s_lab), run_time=0.3)

        # Animate 4 representative anti-diagonals
        for k in [0, 1, 3, 5]:
            highlights = VGroup()
            for i in range(4):
                j = k - i
                if 0 <= j < 4:
                    h = SurroundingRectangle(
                        cells[i][j], color=diag_colors[k],
                        buff=0.03, stroke_width=2.5,
                    )
                    highlights.add(h)
            self.play(Create(highlights), run_time=0.35)
            self.play(
                FadeIn(res_cells[k]), FadeIn(res_labels[k]),
                run_time=0.3,
            )
            self.play(FadeOut(highlights), run_time=0.2)

        # Fill remaining cells at once
        remaining = VGroup()
        remaining_labs = VGroup()
        for k in range(7):
            if k not in [0, 1, 3, 5]:
                remaining.add(res_cells[k])
                remaining_labs.add(res_labels[k])
        self.play(FadeIn(remaining), FadeIn(remaining_labs), run_time=0.4)

        note = Text(
            "Sum along anti-diagonals collects terms for c\u2096",
            font_size=16, color=GRAY_B,
        )
        note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(note), run_time=0.3)
        self.wait(1.5)
        self.wipe()

    # ─────────────────────────────────────────────────────────
    # 8  LAYER 5 — BIT-WEIGHT GROUPING
    # ─────────────────────────────────────────────────────────

    def scene_layer5(self):
        self.header("Layer 5: Channel Grouping (1\u00d71 Conv)")

        # Explain the idea
        idea = Text(
            "Only the sum r = p + q matters (bit-weight 2\u02b3)",
            font_size=20, color=WHITE,
        )
        idea.shift(UP * 2.0)
        self.play(FadeIn(idea), run_time=0.5)

        # Build grouping diagram: boxes for each (p,q) pair, color by r
        pair_boxes = {}
        group_vgroups = {}
        col_x_start = -5.0
        col_spacing = 2.4

        for r in range(5):
            pairs = GROUPS_BY_R[r]
            col_x = col_x_start + r * col_spacing
            color = R_COLORS[r]

            # r label at top
            r_label = Text(f"r = {r}", font_size=16, color=color, weight="BOLD")
            r_label.move_to([col_x, 1.2, 0])

            # weight label
            w_label = Text(f"(\u00d7{2**r})", font_size=12, color=GRAY_B)
            w_label.next_to(r_label, DOWN, buff=0.05)

            grp = VGroup(r_label, w_label)

            for idx, (p, q) in enumerate(pairs):
                box = RoundedRectangle(
                    width=0.9, height=0.4, corner_radius=0.08,
                    stroke_color=color, stroke_width=2,
                )
                box.set_fill(color, opacity=0.15)
                box.move_to([col_x, 0.4 - idx * 0.5, 0])
                label = Text(f"({p},{q})", font_size=14, color=color)
                label.move_to(box)
                pair_vg = VGroup(box, label)
                grp.add(pair_vg)
                pair_boxes[(p, q)] = pair_vg

            group_vgroups[r] = grp

        all_groups = VGroup(*group_vgroups.values())
        self.play(FadeIn(all_groups), run_time=1.0)
        self.wait(0.8)

        # Show the summing formula
        formula = Text(
            "T[k, r]  =  \u03a3 S(p,q)[k]    for p+q = r",
            font_size=20, color=YELLOW,
        )
        formula.shift(DOWN * 1.3)
        self.play(FadeIn(formula), run_time=0.5)
        self.wait(0.5)

        # Concrete example: k=3
        self.play(FadeOut(idea), FadeOut(formula), run_time=0.3)

        example_header = Text(
            "Example: k = 3   (c\u2083 = 30)",
            font_size=20, color=WHITE,
        )
        example_header.shift(DOWN * 1.2)
        self.play(FadeIn(example_header), run_time=0.4)

        # Show T values below the groups
        t_cells = VGroup()
        for r in range(5):
            v = T_K3[r]
            col_x = col_x_start + r * col_spacing
            cell = value_cell(v, cs=0.55, color=R_COLORS[r] if v > 0 else None)
            cell.move_to([col_x, -1.9, 0])
            t_label = Text(f"T[3,{r}]={v}", font_size=12, color=GRAY_B)
            t_label.next_to(cell, DOWN, buff=0.08)
            t_cells.add(VGroup(cell, t_label))

        self.play(FadeIn(t_cells), run_time=0.7)
        self.wait(0.5)

        # Show the weighted sum
        weighted = Text(
            "c\u2083 = 2\u00b71 + 2\u00b72 + 2\u00b74 + 2\u00b78 + 0 = 30",
            font_size=20, color=YELLOW,
        )
        weighted.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(weighted), run_time=0.5)

        # But T values > 1!
        warning = Text(
            "T values can be > 1 \u2192 need carry propagation!",
            font_size=17, color=C_CARRY, weight="BOLD",
        )
        warning.next_to(weighted, UP, buff=0.15)
        self.play(FadeIn(warning), run_time=0.5)
        self.wait(1.5)
        self.wipe()

    # ─────────────────────────────────────────────────────────
    # 9  LAYER 6 — CARRY PROPAGATION
    # ─────────────────────────────────────────────────────────

    def scene_layer6(self):
        self.header("Layer 6: Carry Propagation (Dense + ReLU)")

        # Explain
        explain = VGroup(
            Text("Convert T values to binary, one column at a time:",
                 font_size=18, color=WHITE),
            Text("  bit  = T mod 2          carry = (T \u2212 bit) / 2",
                 font_size=16, color=GRAY_B),
            Text("  carry added to next column \u2192",
                 font_size=16, color=GRAY_B),
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        explain.shift(UP * 1.8 + LEFT * 1.0)
        self.play(FadeIn(explain), run_time=0.6)
        self.wait(0.5)

        # Build the T value row: 5 cells, with r labels below
        cs = 0.7
        cell_spacing = 1.3
        start_x = -2.6
        y_row = 0.2

        t_values = list(T_K3)  # mutable copy: [2, 2, 2, 2, 0]
        t_cell_groups = []  # each is VGroup(sq, txt)
        r_labels = VGroup()

        for r in range(5):
            x = start_x + r * cell_spacing
            v = t_values[r]
            sq = Square(side_length=cs, stroke_color=GRAY_B, stroke_width=1.5)
            sq.set_fill(R_COLORS[r], opacity=0.3)
            sq.move_to([x, y_row, 0])
            txt = Text(str(v), font_size=22, color=WHITE)
            txt.move_to(sq)
            cell_grp = VGroup(sq, txt)
            t_cell_groups.append(cell_grp)

            rl = Text(f"r={r}", font_size=13, color=GRAY_B)
            rl.next_to(sq, DOWN, buff=0.12)
            r_labels.add(rl)

        all_cells_vg = VGroup(*t_cell_groups)
        t_header = Text(
            "T[3, r] for k=3:",
            font_size=16, color=WHITE,
        )
        t_header.next_to(all_cells_vg, UP, buff=0.2)

        self.play(FadeIn(all_cells_vg), FadeIn(r_labels), FadeIn(t_header),
                  run_time=0.6)

        # Bit output row (above the T row, will be filled in)
        bits_label = Text("Output bits:", font_size=14, color=C_OK)
        bits_label.move_to([start_x - 1.5, y_row + 1.2, 0])
        self.play(FadeIn(bits_label), run_time=0.3)

        bit_displays = []
        for r in range(5):
            x = start_x + r * cell_spacing
            bit_sq = Square(side_length=cs * 0.7, stroke_color=GRAY_B,
                            stroke_width=1.0)
            bit_sq.set_fill("#1a1a2e", opacity=0.5)
            bit_sq.move_to([x, y_row + 1.2, 0])
            bit_displays.append(bit_sq)
        self.play(*[FadeIn(b) for b in bit_displays], run_time=0.3)

        # Status text area (below the T row)
        status_y = y_row - 1.2

        # Animate carry propagation step by step
        for step_idx, (r, t_before, bit, carry, t_next) in enumerate(CARRY_STEPS):
            x = start_x + r * cell_spacing

            # Highlight current cell
            highlight = SurroundingRectangle(
                t_cell_groups[r], color=YELLOW, buff=0.06, stroke_width=3,
            )
            self.play(Create(highlight), run_time=0.25)

            # Show status
            status_parts = VGroup(
                Text(f"T = {t_before}", font_size=16, color=WHITE),
                Text(f"bit = {t_before} mod 2 = {bit}",
                     font_size=16, color=C_OK),
            )
            if carry > 0:
                status_parts.add(
                    Text(f"carry = ({t_before}\u2212{bit})/2 = {carry}",
                         font_size=16, color=C_CARRY),
                )
            status_parts.arrange(DOWN, buff=0.1, aligned_edge=LEFT)
            status_parts.move_to([1.5, status_y, 0])
            self.play(FadeIn(status_parts), run_time=0.35)

            # Fill in the bit
            bit_txt = Text(str(bit), font_size=18,
                           color=C_OK if bit == 1 else GRAY_C)
            bit_txt.move_to(bit_displays[r])
            bit_fill_color = C_ONE if bit == 1 else C_ZERO
            self.play(
                bit_displays[r].animate.set_fill(bit_fill_color, opacity=0.8),
                FadeIn(bit_txt),
                run_time=0.3,
            )

            # Show carry arrow and update next cell
            carry_arrow = None
            if carry > 0 and r < 4:
                x_next = start_x + (r + 1) * cell_spacing
                carry_arrow = Arrow(
                    [x + cs / 2 + 0.05, y_row, 0],
                    [x_next - cs / 2 - 0.05, y_row, 0],
                    color=C_CARRY, stroke_width=3, buff=0.0,
                    max_tip_length_to_length_ratio=0.25,
                )
                carry_label = Text(
                    f"+{carry}", font_size=14, color=C_CARRY, weight="BOLD",
                )
                carry_label.next_to(carry_arrow, UP, buff=0.03)

                self.play(GrowArrow(carry_arrow), FadeIn(carry_label),
                          run_time=0.3)

                # Update next cell's text
                old_txt = t_cell_groups[r + 1][1]
                new_txt = Text(str(t_next), font_size=22, color=WHITE)
                new_txt.move_to(t_cell_groups[r + 1][0])
                self.play(
                    FadeOut(old_txt), FadeIn(new_txt),
                    run_time=0.25,
                )
                t_cell_groups[r + 1] = VGroup(t_cell_groups[r + 1][0], new_txt)

                self.play(FadeOut(carry_arrow), FadeOut(carry_label),
                          run_time=0.15)

            self.play(FadeOut(highlight), FadeOut(status_parts), run_time=0.2)

        # Final result
        result_bits_str = "".join(str(b) for b in CARRY_RESULT_BITS)
        result_text = Text(
            f"Bits: [{result_bits_str}] \u2192 "
            f"0+2+4+8+16 = 30 = c\u2083  \u2713",
            font_size=18, color=C_OK, weight="BOLD",
        )
        result_text.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(result_text), run_time=0.5)

        # Mod 2 note
        mod_note = Text(
            "mod 2 computed via sawtooth: ReLU(x) \u2212 2\u00b7ReLU(x\u22121) "
            "+ 2\u00b7ReLU(x\u22122) \u2212 \u2026",
            font_size=14, color=GRAY_B,
        )
        mod_note.next_to(result_text, UP, buff=0.15)
        self.play(FadeIn(mod_note), run_time=0.4)
        self.wait(2)
        self.wipe()

    # ─────────────────────────────────────────────────────────
    # 10  FINAL RESULT
    # ─────────────────────────────────────────────────────────

    def scene_result(self):
        self.header("Output: Product Polynomial")

        og, _ = build_grid(OUT_BITS, cs=0.45)
        og.shift(LEFT * 2.5 + DOWN * 0.3)
        ol = Text("Output: 7 \u00d7 5 binary", font_size=16, color=GRAY_B)
        ol.next_to(og, UP, buff=0.15)
        self.play(FadeIn(og), FadeIn(ol), run_time=0.6)

        # Decoded coefficients
        dec = VGroup()
        for k, c in enumerate(PRODUCT):
            r = Text(
                f"c{str(k).translate(SUB)} = {c}",
                font_size=22, color=WHITE,
            )
            dec.add(r)
        dec.arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        dec.shift(RIGHT * 2.5 + DOWN * 0.3)
        dl = Text("Decoded:", font_size=16, color=GRAY_B)
        dl.next_to(dec, UP, buff=0.15)

        self.play(FadeIn(dl), FadeIn(dec), run_time=0.7)
        self.wait(0.8)

        # Product polynomial
        result = Text(
            "R(x) = 3 + 17x + 31x\u00b2 + 30x\u00b3 "
            "+ 20x\u2074 + 7x\u2075 + 2x\u2076",
            font_size=24, color=YELLOW,
        )
        result.to_edge(DOWN, buff=0.9)
        check = Text(
            "\u2713  Exact match with numpy.convolve",
            font_size=16, color=C_OK,
        )
        check.next_to(result, DOWN, buff=0.2)

        self.play(Write(result), run_time=1.0)
        self.play(FadeIn(check), run_time=0.4)
        self.wait(1.5)
        self.wipe()

        # Closing card
        final = Text(
            "Exact polynomial product\ncomputed entirely with CNN layers",
            font_size=38, weight="BOLD", line_spacing=1.3,
        )
        final.set_color_by_gradient(BLUE_B, TEAL_B)
        ops = Text(
            "Reshape  \u00b7  Broadcast  \u00b7  ReLU  \u00b7  Dense  \u00b7  Pooling",
            font_size=20, color=GRAY_B,
        )
        ops.next_to(final, DOWN, buff=0.5)

        self.play(Write(final), run_time=1.5)
        self.play(FadeIn(ops), run_time=0.5)
        self.wait(2.5)
