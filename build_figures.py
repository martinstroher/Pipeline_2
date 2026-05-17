"""
build_figures.py — Generate presentation figures for ICEIS 2026 slides.

Run:  python build_figures.py [fig_number|all]

Outputs to docs/figures/ as PNGs at 300 DPI.

Figures:
  1 — Upper Ontology Stack (slide 2)
  2 — NLD Anatomy (slide 3)
  3 — Pipeline Flowchart 4 Phases (slide 4)
  4 — Four Pipelines Comparison (slide 5)
  5 — Classification Waterfall (slide 6)
  6 — Expert Evaluation Design (slide 7)
  7 — Taxonomy Tree (slide 10)
  9 — Extended Pipeline Architecture (slide 12)
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────────────────────────
# STYLE / PALETTE — consistent with presentation
# ─────────────────────────────────────────────────────────────────────────────
COL_TITLE      = "#1F3864"
COL_BLUE       = "#2E75B6"
COL_LIGHTBLUE  = "#9DC3E6"
COL_PALEBLUE   = "#D9E2F3"
COL_ORANGE     = "#ED7D31"
COL_GREEN      = "#1E8855"
COL_RED        = "#C0392B"
COL_GOLD       = "#DAA520"
COL_GRAY       = "#555555"
COL_LIGHTGRAY  = "#BFBFBF"
COL_TEXT       = "#1A1A2E"
COL_WHITE      = "#FFFFFF"

OUT = "docs/figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11


def _save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved → {path}")


def _box(ax, x, y, w, h, color, text, text_color="white",
        fontsize=11, bold=True, alpha=1.0, rounded=0.02, edgecolor=None):
    """Helper: rounded box with centered text."""
    ec = edgecolor if edgecolor else color
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0.005,rounding_size={rounded}",
                          linewidth=1.2, edgecolor=ec, facecolor=color, alpha=alpha)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center",
            color=text_color, fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            wrap=True)


def _arrow(ax, x1, y1, x2, y2, color=COL_GRAY, lw=2.5, style="->,head_width=8,head_length=10"):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                         arrowstyle=style, color=color, linewidth=lw,
                         shrinkA=2, shrinkB=2)
    ax.add_patch(a)


# ═════════════════════════════════════════════════════════════════════════════
# FIG 1 — UPPER ONTOLOGY STACK
# ═════════════════════════════════════════════════════════════════════════════
def fig1_ontology_stack():
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(6.5, 9.5, "Upper Ontology Cascade",
            ha="center", fontsize=18, fontweight="bold", color=COL_TITLE)

    # Each layer occupies full width; pyramid metaphor encoded by color saturation
    # (light → dark) and indentation on the left edge.
    # Layer: (y, indent, color, title_color, title, subtitle, chips)
    layers = [
        (7.5, 0.5, COL_PALEBLUE,  COL_TITLE, "BFO",
            "Basic Formal Ontology — universal categories",
            ["Continuant", "Process", "Quality"]),
        (5.7, 1.3, COL_LIGHTBLUE, COL_TITLE, "GeoCore",
            "General geological science",
            ["Rock", "Geological Structure", "Diagenetic Process"]),
        (3.9, 2.1, COL_BLUE,      "white",   "GeoReservoir",
            "Petroleum-reservoir-specific",
            ["Porosity Type", "Reservoir Seal", "Hydrocarbon Column"]),
        (2.1, 2.9, COL_ORANGE,    "white",   "Pre-Salt domain terms",
            "Anchored at most specific level",
            ["Coquina", "Stromatolite", "Dolomitization"]),
    ]

    right_edge = 12.5  # All layers end at the same right edge

    for y, indent, color, tc, title, subtitle, chips in layers:
        x = indent
        w = right_edge - x
        # Main band
        ax.add_patch(FancyBboxPatch((x, y), w, 1.3,
                                      boxstyle="round,pad=0.01,rounding_size=0.05",
                                      linewidth=0, facecolor=color))
        # Title and subtitle on the LEFT side of band
        ax.text(x + 0.25, y + 0.92, title,
                fontsize=14, fontweight="bold", color=tc, va="center")
        ax.text(x + 0.25, y + 0.42, subtitle,
                fontsize=9.5, color=tc, va="center", alpha=0.95)

        # Chips on the RIGHT side — distributed evenly in a fixed right zone
        chips_zone_left = x + w * 0.50
        chips_zone_right = x + w - 0.2
        n_chips = len(chips)
        slot_w = (chips_zone_right - chips_zone_left) / n_chips
        for i, chip in enumerate(chips):
            cx = chips_zone_left + slot_w * (i + 0.5)
            ax.text(cx, y + 0.65, chip,
                    fontsize=9, color=tc, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.35",
                              facecolor="white", edgecolor=tc, alpha=0.85, linewidth=0.7))

    # "Increasing specificity" arrow on the right (just inside figure margin)
    ax.annotate("", xy=(12.8, 2.3), xytext=(12.8, 8.5),
                arrowprops=dict(arrowstyle="->,head_width=0.5,head_length=0.7",
                                color=COL_GRAY, linewidth=2.5))
    ax.text(12.95, 5.4, "increasing specificity",
            fontsize=10, color=COL_GRAY, rotation=-90, va="center", style="italic")

    # Waterfall callout bottom
    ax.text(6.5, 0.7,
            "Waterfall classification:  GeoReservoir  →  GeoCore  →  BFO",
            ha="center", fontsize=12, color=COL_TITLE, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0F4FA",
                      edgecolor=COL_TITLE, linewidth=1.2))

    _save(fig, "fig1_ontology_stack.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 2 — NLD ANATOMY ("X is a Y that Z")
# ═════════════════════════════════════════════════════════════════════════════
def fig2_nld_anatomy():
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Title
    ax.text(8, 7.4, 'Natural Language Definition — Aristotelian Form',
            ha="center", fontsize=18, fontweight="bold", color=COL_TITLE)
    ax.text(8, 6.85, '"X  is a  Y  that  Z"',
            ha="center", fontsize=14, color=COL_GRAY, style="italic")

    # Render sentence by computing each part's text extent and placing them sequentially.
    # Each part: (text, color, is_role)
    fontsize = 13.5
    parts = [
        ("Dolomitization",                                     COL_BLUE,   True,  "X — definiendum",     "the term being defined"),
        ("is a",                                                COL_GRAY,   False, None, None),
        ("diagenetic process",                                  COL_ORANGE, True,  "Y — proximate genus", "parent class →\nbecomes a taxonomy node"),
        ("that",                                                COL_GRAY,   False, None, None),
        ("replaces calcite by dolomite via Mg-rich fluids",     COL_GREEN,  True,  "Z — differentia",     "distinguishing characteristic"),
    ]

    text_y = 4.6
    underline_y = 4.25
    role_y = 3.75
    caption_y = 3.10

    # Use renderer to measure text width in data units
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    # Build a temporary text artist to measure each part's width in data coords
    def measure_width(text, fs, weight):
        t = ax.text(0, 0, text, fontsize=fs, fontweight=weight, alpha=0)
        bbox_pix = t.get_window_extent(renderer=renderer)
        bbox_data = inv.transform(bbox_pix)
        w = bbox_data[1, 0] - bbox_data[0, 0]
        t.remove()
        return w

    # Measure widths
    measured = []
    for text, color, is_role, _, _ in parts:
        w = measure_width(text, fontsize, "bold" if is_role else "normal")
        measured.append(w)

    # Compute total + gaps, then center on x=8
    inner_gap = 0.35
    total_width = sum(measured) + inner_gap * (len(parts) - 1)
    start_x = 8 - total_width / 2

    cursor = start_x
    role_data = []  # (color, label, caption, center_x)
    for (text, color, is_role, role_label, caption), w in zip(parts, measured):
        cx = cursor + w / 2
        # The sentence text
        ax.text(cx, text_y, text,
                ha="center", va="center", fontsize=fontsize, color=COL_TEXT,
                fontweight="bold" if is_role else "normal")
        # Underline (only for X, Y, Z)
        if is_role:
            ax.add_patch(Rectangle((cursor, underline_y), w, 0.08,
                                    facecolor=color, edgecolor="none"))
            role_data.append((color, role_label, caption, cx))
        cursor += w + inner_gap

    # Role labels + captions under the colored parts
    for color, label, caption, cx in role_data:
        ax.text(cx, role_y, label,
                ha="center", fontsize=11.5, color=color, fontweight="bold")
        ax.text(cx, caption_y, caption,
                ha="center", va="top", fontsize=10, color=COL_GRAY, style="italic")

    # Bottom: "why this matters" panel
    box_y = 0.3
    box_h = 1.6
    ax.add_patch(FancyBboxPatch((0.5, box_y), 15, box_h,
                                  boxstyle="round,pad=0.02,rounding_size=0.1",
                                  facecolor="#FFF8E7", edgecolor=COL_GOLD, linewidth=1.5))

    ax.text(0.95, box_y + box_h - 0.35, "Why it matters",
            fontsize=11, fontweight="bold", color=COL_TITLE)
    ax.text(0.95, box_y + 0.85,
            "Explicit genus  →  taxonomy node:",
            fontsize=11, color=COL_TEXT)

    # Mini taxonomy
    px = 5.4
    py = box_y + 0.7
    ax.add_patch(FancyBboxPatch((px, py), 2.5, 0.55,
                                  boxstyle="round,pad=0.02,rounding_size=0.08",
                                  facecolor=COL_ORANGE, edgecolor="none"))
    ax.text(px + 1.25, py + 0.275, "Diagenetic Process",
            ha="center", va="center", fontsize=10, color="white", fontweight="bold")

    # Single arrow from parent to a junction, then horizontal "umbrella" line over children
    junction_x = 8.4
    children_y = py + 0.275
    _arrow(ax, px + 2.5, children_y, junction_x - 0.05, children_y,
           color=COL_ORANGE, lw=1.8, style="->,head_width=5,head_length=7")

    # Children boxes side by side (no inter-child arrows)
    children = ["Dolomitization", "Silicification", "Cementation"]
    cx_start = 8.5
    child_w = 1.85
    child_gap = 0.18
    # Horizontal connector line across all children tops
    last_cx = cx_start + (len(children) - 1) * (child_w + child_gap)
    ax.plot([cx_start, last_cx + child_w], [children_y + 0.32, children_y + 0.32],
            color=COL_ORANGE, linewidth=1.2)
    for i, child in enumerate(children):
        cx = cx_start + i * (child_w + child_gap)
        # vertical drop from connector to child top
        ax.plot([cx + child_w / 2, cx + child_w / 2],
                [children_y + 0.32, children_y + 0.25],
                color=COL_ORANGE, linewidth=1.2)
        ax.add_patch(FancyBboxPatch((cx, children_y - 0.225), child_w, 0.45,
                                      boxstyle="round,pad=0.02,rounding_size=0.06",
                                      facecolor="white", edgecolor=COL_ORANGE, linewidth=1.2))
        ax.text(cx + child_w / 2, children_y, child,
                ha="center", va="center", fontsize=9, color=COL_ORANGE, fontweight="bold")

    ax.text(0.95, box_y + 0.3,
            "→ Same genus shared across terms = automatic intermediate class in the hierarchy.",
            fontsize=9.5, color=COL_GRAY, style="italic")

    _save(fig, "fig2_nld_anatomy.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 3 — PIPELINE FLOWCHART (4 phases)
# ═════════════════════════════════════════════════════════════════════════════
def fig3_pipeline_4phases():
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(7.5, 5.6, "Methodology — 4-Phase Pipeline",
            ha="center", fontsize=18, fontweight="bold", color=COL_TITLE)

    phases = [
        ("1", "CORPUS",        COL_GRAY,    "40 peer-reviewed\nPre-Salt articles\nSpecialist-curated",     "papers"),
        ("2", "EXTRACTION",    COL_BLUE,    "4 pipelines:\nLLM+Corpus / LLM-Gen\nNER+Corpus / TF-IDF",     "term lists"),
        ("3", "CLASSIFICATION", COL_ORANGE, "Aristotelian NLDs\nCoT categorization\nBFO → GeoCore → GeoR.", "categorized terms"),
        ("4", "VALIDATION",    COL_GREEN,   "5 domain experts\n100 reference terms\nBlind evaluation",     "validated ontology"),
    ]

    card_w, card_h = 2.9, 3.0
    gap = 0.35
    start_x = (15 - (4 * card_w + 3 * gap)) / 2
    y = 1.4

    for i, (num, title, color, body, arrow_label) in enumerate(phases):
        x = start_x + i * (card_w + gap)
        # Card background
        ax.add_patch(FancyBboxPatch((x, y), card_w, card_h,
                                      boxstyle="round,pad=0.01,rounding_size=0.1",
                                      facecolor="white", edgecolor=COL_LIGHTGRAY, linewidth=1.2))
        # Colored top stripe
        ax.add_patch(FancyBboxPatch((x, y + card_h - 0.7), card_w, 0.7,
                                      boxstyle="round,pad=0.01,rounding_size=0.1",
                                      facecolor=color, edgecolor="none"))
        # Cover bottom of stripe (square corners on bottom side)
        ax.add_patch(Rectangle((x + 0.02, y + card_h - 0.85), card_w - 0.04, 0.15,
                                facecolor=color, edgecolor="none"))

        # Phase number circle
        circle = plt.Circle((x + 0.35, y + card_h - 0.35), 0.22,
                             facecolor="white", edgecolor=color, linewidth=2)
        ax.add_patch(circle)
        ax.text(x + 0.35, y + card_h - 0.35, num,
                ha="center", va="center", fontsize=12, fontweight="bold", color=color)

        # Title
        ax.text(x + card_w / 2 + 0.1, y + card_h - 0.35, title,
                ha="center", va="center", fontsize=13, fontweight="bold", color="white")

        # Body
        ax.text(x + card_w / 2, y + 1.3, body,
                ha="center", va="center", fontsize=10.5, color=COL_TEXT, linespacing=1.5)

        # Arrow to next card
        if i < 3:
            ax_x1 = x + card_w
            ax_x2 = x + card_w + gap
            _arrow(ax, ax_x1, y + card_h / 2, ax_x2, y + card_h / 2,
                   color=color, lw=3.5,
                   style="->,head_width=8,head_length=10")

    # Final output label after card 4
    ax.text(start_x + 4 * card_w + 3 * gap + 0.1, y + card_h / 2, "→ ontology",
            ha="left", va="center", fontsize=9, color=COL_GREEN, fontweight="bold")

    _save(fig, "fig3_pipeline_4phases.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 4 — FOUR PIPELINES COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
def fig4_four_pipelines():
    fig, ax = plt.subplots(figsize=(16, 5.5))
    ax.set_xlim(0, 16)
    ax.set_ylim(3.5, 9)
    ax.axis("off")

    ax.text(8, 8.5, "Four Term-Extraction Pipelines — Side-by-Side",
            ha="center", fontsize=18, fontweight="bold", color=COL_TITLE)

    pipelines = [
        ("P1 — LLM + Corpus",   COL_BLUE,
            "□  40 papers",
            "LLM reads each paper,\nextracts terms in-context",
            "Top 100\nby frequency",
            ("Grounded in source", "API cost")),
        ("P2 — LLM Generated",  COL_ORANGE,
            "⚪  (no corpus)",
            "LLM recalls\nfrom parametric memory",
            "Top 100\nzero-shot",
            ("Tests model recall", "Hallucination risk")),
        ("P3 — NER + Corpus",   COL_GOLD,
            "□  40 papers",
            "Extracts named entities\nusing NER model specialized\nin petroleum corpus",
            "Top 100\nentity tags",
            ("Fast, deterministic", "Generic — no domain")),
        ("P4 — TF-IDF",         COL_GRAY,
            "□  40 papers",
            "Word frequency\n× inverse doc. freq.",
            "Top 100\nby score",
            ("Traditional baseline", "No semantics")),
    ]

    col_w = 3.5
    gap = 0.25
    start_x = (16 - (4 * col_w + 3 * gap)) / 2
    top_y = 7.5
    row_h = 1.25
    row_labels = ["INPUT", "METHOD", "OUTPUT"]
    row_ys = [top_y - row_h, top_y - 2 * row_h, top_y - 3 * row_h]

    # Row labels on the left
    for label, ry in zip(row_labels, row_ys):
        ax.text(start_x - 0.25, ry + row_h / 2, label,
                ha="right", va="center", fontsize=10, fontweight="bold", color=COL_GRAY)

    for i, (title, color, inp, method, output, (pro, con)) in enumerate(pipelines):
        x = start_x + i * (col_w + gap)
        # Header band
        ax.add_patch(FancyBboxPatch((x, top_y), col_w, 0.6,
                                      boxstyle="round,pad=0.01,rounding_size=0.06",
                                      facecolor=color, edgecolor="none"))
        ax.text(x + col_w / 2, top_y + 0.3, title,
                ha="center", va="center", fontsize=12, fontweight="bold", color="white")

        # Tinted background for the column body
        body_bottom = top_y - 3 * row_h
        ax.add_patch(Rectangle((x, body_bottom), col_w, 3 * row_h,
                                facecolor=color, alpha=0.06, edgecolor=color, linewidth=0.5))

        # Row separators
        for j in range(1, 3):
            ry = top_y - j * row_h
            ax.plot([x, x + col_w], [ry, ry], color=COL_LIGHTGRAY, linewidth=0.7)

        # Content per row
        contents = [inp, method, output]
        for j, content in enumerate(contents):
            cy = top_y - j * row_h - row_h / 2
            ax.text(x + col_w / 2, cy, content,
                    ha="center", va="center", fontsize=10, color=COL_TEXT, linespacing=1.4)

    _save(fig, "fig4_four_pipelines.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 5 — CLASSIFICATION WATERFALL
# ═════════════════════════════════════════════════════════════════════════════
def fig5_waterfall():
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(7.5, 8.5, "Classification Waterfall — Most Specific Match Wins",
            ha="center", fontsize=18, fontweight="bold", color=COL_TITLE)

    # ─── PANEL A: Generic waterfall (left) ───
    ax.text(2.8, 7.8, "Decision Cascade",
            ha="center", fontsize=13, fontweight="bold", color=COL_TITLE)

    # Input box
    ax.add_patch(FancyBboxPatch((1.0, 6.7), 3.6, 0.7,
                                  boxstyle="round,pad=0.02,rounding_size=0.08",
                                  facecolor="#F0F4FA", edgecolor=COL_TITLE, linewidth=1.5))
    ax.text(2.8, 7.05, "Input:  Term + NLD",
            ha="center", va="center", fontsize=11, fontweight="bold", color=COL_TITLE)

    # Decision boxes (waterfall) — confined to LEFT panel (x < 9.0)
    decisions = [
        (5.65, "Fits GeoReservoir?",  COL_BLUE,      "geo:HydrocarbonColumn"),
        (4.40, "Fits GeoCore?",       COL_LIGHTBLUE, "geo:SedimentaryRock"),
        (3.15, "Fits BFO?",           COL_PALEBLUE,  "bfo:Process"),
    ]
    nc_y = 1.90  # NOT_CLASSIFIED y

    for y, q, color, accept in decisions:
        # Decision box
        tc = "white" if color in (COL_BLUE, COL_TITLE) else COL_TITLE
        ax.add_patch(FancyBboxPatch((1.0, y), 3.6, 0.85,
                                      boxstyle="round,pad=0.02,rounding_size=0.08",
                                      facecolor=color, edgecolor=COL_TITLE, linewidth=1.2))
        ax.text(2.8, y + 0.42, q,
                ha="center", va="center", fontsize=12, fontweight="bold", color=tc)

        # YES → accept (right) — stays in left panel
        _arrow(ax, 4.65, y + 0.42, 5.3, y + 0.42, color=COL_GREEN, lw=2)
        ax.text(4.95, y + 0.6, "yes", ha="center", fontsize=9, color=COL_GREEN, fontweight="bold")
        # Accept label — fits within left panel (x ends ~9.0)
        ax.text(5.4, y + 0.42, f"✓  {accept}",
                ha="left", va="center", fontsize=10.5, color=COL_GREEN, fontweight="bold")

    # Arrows from box to box (no path on YES side)
    # input → first decision
    _arrow(ax, 2.8, 6.65, 2.8, 6.55, color=COL_GRAY, lw=2)
    # cascading "no" arrows
    cascade_ys = [(decisions[0][0], decisions[1][0]),
                  (decisions[1][0], decisions[2][0]),
                  (decisions[2][0], nc_y)]
    for y_from, y_to in cascade_ys:
        _arrow(ax, 2.8, y_from, 2.8, y_to + 0.85, color=COL_GRAY, lw=2)
        ax.text(3.0, (y_from + y_to + 0.85) / 2, "no",
                ha="left", fontsize=9, color=COL_GRAY, style="italic")

    # NOT_CLASSIFIED terminal
    ax.add_patch(FancyBboxPatch((1.0, nc_y), 3.6, 0.7,
                                  boxstyle="round,pad=0.02,rounding_size=0.08",
                                  facecolor=COL_LIGHTGRAY, edgecolor=COL_GRAY, linewidth=1.5))
    ax.text(2.8, nc_y + 0.35, "NOT_CLASSIFIED",
            ha="center", va="center", fontsize=11, fontweight="bold", color=COL_GRAY)
    ax.text(5.4, nc_y + 0.35, "✗  out of scope",
            ha="left", va="center", fontsize=10.5, color=COL_RED, fontweight="bold")

    # ─── PANEL B: Worked example (right) ───
    # Vertical divider
    ax.plot([9.5, 9.5], [1.3, 7.8], color=COL_LIGHTGRAY, linewidth=1.2, linestyle="--")

    ax.text(12.25, 7.8, "Worked Example",
            ha="center", fontsize=13, fontweight="bold", color=COL_TITLE)

    # Input NLD
    ax.add_patch(FancyBboxPatch((9.9, 6.4), 4.7, 1.1,
                                  boxstyle="round,pad=0.02,rounding_size=0.08",
                                  facecolor="#FFF8E7", edgecolor=COL_GOLD, linewidth=1.5))
    ax.text(12.25, 7.15, "Term: Dolomitization",
            ha="center", fontsize=10.5, fontweight="bold", color=COL_TITLE)
    ax.text(12.25, 6.7, '"a diagenetic process that\nreplaces calcite by dolomite..."',
            ha="center", va="center", fontsize=9, color=COL_TEXT, style="italic")

    # Trace
    trace_steps = [
        (5.6, "GeoReservoir?",  COL_BLUE,      "✗  not reservoir-specific",   COL_RED),
        (4.2, "GeoCore?",       COL_LIGHTBLUE, "✓  DiageneticProcess",        COL_GREEN),
    ]
    _arrow(ax, 12.25, 6.35, 12.25, 6.15, color=COL_GRAY, lw=2)
    for y, label, color, result, rcolor in trace_steps:
        # Step box
        tc = "white" if color == COL_BLUE else COL_TITLE
        ax.add_patch(FancyBboxPatch((9.9, y), 4.7, 0.75,
                                      boxstyle="round,pad=0.02,rounding_size=0.06",
                                      facecolor=color, edgecolor=COL_TITLE, linewidth=1.2))
        ax.text(11.1, y + 0.375, label,
                ha="center", va="center", fontsize=11, fontweight="bold", color=tc)
        ax.text(12.65, y + 0.375, result,
                ha="left", va="center", fontsize=10, color=rcolor, fontweight="bold")
    # arrow between steps
    _arrow(ax, 12.25, 5.55, 12.25, 4.98, color=COL_GRAY, lw=2)

    # Final result
    final_y = 2.6
    _arrow(ax, 12.25, 4.15, 12.25, final_y + 0.75, color=COL_GREEN, lw=2.5)
    ax.add_patch(FancyBboxPatch((9.9, final_y), 4.7, 0.7,
                                  boxstyle="round,pad=0.02,rounding_size=0.08",
                                  facecolor=COL_GREEN, edgecolor=COL_GREEN, linewidth=1.5))
    ax.text(12.25, final_y + 0.35, "→  geo:DiageneticProcess",
            ha="center", va="center", fontsize=12, fontweight="bold", color="white")
    ax.text(12.25, final_y - 0.35, "Classified at most specific applicable level",
            ha="center", fontsize=9, color=COL_GRAY, style="italic")

    # Bottom annotation
    ax.text(7.5, 0.6,
            "Cascade ensures each term is classified at the most specific applicable level — "
            "BFO is only used as a fallback for cross-domain abstract categories.",
            ha="center", fontsize=10, color=COL_GRAY, style="italic")

    _save(fig, "fig5_waterfall.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 6 — EXPERT EVALUATION DESIGN
# ═════════════════════════════════════════════════════════════════════════════
def fig6_expert_eval():
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(8, 6.5, "Expert Evaluation Design — Blinded Scoring",
            ha="center", fontsize=18, fontweight="bold", color=COL_TITLE)

    stages = [
        ("POOL",        COL_BLUE,    "4 × 100\n= 400 entries",                "All pipeline outputs"),
        ("BLINDING",    COL_ORANGE,  "Shuffle\n+ anonymize",                  "Origin hidden"),
        ("ASSIGNMENT",  COL_GOLD,    "5 experts\n20 terms each",              "PhD / MSc, ≤20 yrs"),
        ("SCORING",     COL_GREEN,   "3 criteria\nper term",                  "see below"),
        ("AGGREGATION", COL_TITLE,   "Consensus\nscoring",                    "Strict / Majority"),
    ]

    card_w, card_h = 2.6, 2.3
    gap = 0.5
    start_x = (16 - (5 * card_w + 4 * gap)) / 2
    y = 2.5

    for i, (title, color, body, footer) in enumerate(stages):
        x = start_x + i * (card_w + gap)
        ax.add_patch(FancyBboxPatch((x, y), card_w, card_h,
                                      boxstyle="round,pad=0.01,rounding_size=0.1",
                                      facecolor="white", edgecolor=color, linewidth=2))
        # Header band
        ax.add_patch(FancyBboxPatch((x, y + card_h - 0.55), card_w, 0.55,
                                      boxstyle="round,pad=0.01,rounding_size=0.1",
                                      facecolor=color, edgecolor="none"))
        ax.add_patch(Rectangle((x + 0.02, y + card_h - 0.65), card_w - 0.04, 0.1,
                                facecolor=color, edgecolor="none"))
        ax.text(x + card_w / 2, y + card_h - 0.275, title,
                ha="center", va="center", fontsize=12, fontweight="bold", color="white")
        # Body
        ax.text(x + card_w / 2, y + card_h - 1.25, body,
                ha="center", va="center", fontsize=11, color=COL_TEXT, linespacing=1.5, fontweight="bold")
        # Footer
        ax.text(x + card_w / 2, y + 0.4, footer,
                ha="center", va="center", fontsize=9, color=COL_GRAY, style="italic")

        # Arrow to next
        if i < 4:
            _arrow(ax, x + card_w, y + card_h / 2,
                   x + card_w + gap, y + card_h / 2,
                   color=COL_GRAY, lw=2.5,
                   style="->,head_width=7,head_length=9")

    # Scoring criteria detail at the bottom
    crit_y = 0.4
    crit_h = 1.3
    ax.add_patch(FancyBboxPatch((0.5, crit_y), 15, crit_h,
                                  boxstyle="round,pad=0.02,rounding_size=0.1",
                                  facecolor="#F8FAFC", edgecolor=COL_LIGHTGRAY, linewidth=1.0))
    ax.text(1.0, crit_y + crit_h - 0.25, "Scoring criteria",
            fontsize=10.5, fontweight="bold", color=COL_TITLE)

    criteria = [
        ("Term Relevance",     "Relevant  /  Irrelevant  /  Invalid  /  Unknown",                                        COL_BLUE),
        ("NLD Accuracy",       "Correct  /  Partial  /  Incorrect  /  Ambiguous  /  Wrong context  /  Unknown",           COL_ORANGE),
        ("Category Accuracy",  "Correct & Specific  /  Correct but Unspecific  /  Incorrect  /  Unknown",                 COL_GREEN),
    ]
    for i, (name, opts, color) in enumerate(criteria):
        cy = crit_y + crit_h - 0.55 - i * 0.28
        ax.text(1.4, cy, "●", color=color, fontsize=14, va="center")
        ax.text(1.7, cy, f"{name}:",
                fontsize=10, fontweight="bold", color=COL_TEXT, va="center")
        ax.text(4.4, cy, opts,
                fontsize=9.5, color=COL_GRAY, va="center")

    _save(fig, "fig6_expert_eval.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 7 — TAXONOMY TREE (real data from 7_ontology.ttl)
# ═════════════════════════════════════════════════════════════════════════════
def fig7_taxonomy_tree():
    """
    Horizontal tree showing a snippet of the produced ontology.
    Uses representative branches across BFO / GeoCore / GeoReservoir.
    """
    fig, ax = plt.subplots(figsize=(15, 9.5))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(7.5, 9.5, "Ontology Snippet — Pre-Salt Taxonomy Excerpt",
            ha="center", fontsize=18, fontweight="bold", color=COL_TITLE)
    ax.text(7.5, 9.05, "Anchored to BFO  →  GeoCore  →  GeoReservoir",
            ha="center", fontsize=11, color=COL_GRAY, style="italic")

    # Node definitions: (x, y, label, color, fontcolor)
    # Three branches stacked vertically
    nodes = {
        # Root layer (upper ontologies) on the far left
        "BFO":           (0.8, 2.0, "BFO",           COL_PALEBLUE,  COL_TITLE),
        "GeoCore":       (0.8, 5.5, "GeoCore",       COL_LIGHTBLUE, COL_TITLE),
        "GeoReservoir":  (0.8, 8.0, "GeoReservoir",  COL_BLUE,      "white"),

        # GeoReservoir branch (top)
        "PorosityType":      (5.0, 8.4, "Porosity Type",     COL_BLUE, "white"),
        "ReservoirSeal":     (5.0, 7.6, "Reservoir Seal",    COL_BLUE, "white"),
        "VuggyPorosity":     (9.0, 8.7, "Vuggy Porosity",    COL_ORANGE, "white"),
        # "InterparticleP":    (9.0, 8.1, "Interparticle Porosity", COL_ORANGE, "white"),
        "InterparticleP":    (9.2, 8.1, "Interparticle Pore",   COL_ORANGE, "white"),

        # GeoCore branch (middle)
        "SedimentaryRock":   (5.0, 6.2, "Sedimentary Rock",   COL_LIGHTBLUE, COL_TITLE),
        "DiageneticProc":    (5.0, 5.5, "Diagenetic Process", COL_LIGHTBLUE, COL_TITLE),
        "GeologicalStruct":  (5.0, 4.8, "Geological Structure", COL_LIGHTBLUE, COL_TITLE),
        "Coquina":           (9.0, 6.7, "Coquina",            COL_ORANGE, "white"),
        "Grainstone":        (9.0, 6.1, "Grainstone",         COL_ORANGE, "white"),
        "Stromatolite":      (9.0, 5.5, "Stromatolite",       COL_ORANGE, "white"),
        "Dolomitization":    (9.0, 4.9, "Dolomitization",     COL_ORANGE, "white"),
        "Silicification":    (9.0, 4.3, "Silicification",     COL_ORANGE, "white"),

        # BFO branch (bottom)
        "Continuant":        (5.0, 2.4, "Continuant",         COL_PALEBLUE, COL_TITLE),
        "Process":           (5.0, 1.6, "Process",            COL_PALEBLUE, COL_TITLE),
        "Quality":           (5.0, 0.9, "Quality",            COL_PALEBLUE, COL_TITLE),
    }

    # Edges: (parent, child)
    edges = [
        ("GeoReservoir", "PorosityType"),
        ("GeoReservoir", "ReservoirSeal"),
        ("PorosityType", "VuggyPorosity"),
        ("PorosityType", "InterparticleP"),
        ("GeoCore", "SedimentaryRock"),
        ("GeoCore", "DiageneticProc"),
        ("GeoCore", "GeologicalStruct"),
        ("SedimentaryRock", "Coquina"),
        ("SedimentaryRock", "Grainstone"),
        ("SedimentaryRock", "Stromatolite"),
        ("DiageneticProc", "Dolomitization"),
        ("DiageneticProc", "Silicification"),
        ("BFO", "Continuant"),
        ("BFO", "Process"),
        ("BFO", "Quality"),
    ]

    # Draw edges first
    for parent, child in edges:
        px, py, *_ = nodes[parent]
        cx, cy, *_ = nodes[child]
        # Use arrowed connector to make hierarchy direction clear
        ax.annotate("", xy=(cx - 0.95, cy), xytext=(px + 0.95, py),
                    arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.5",
                                    color=COL_LIGHTGRAY, linewidth=1.2,
                                    shrinkA=0, shrinkB=0), zorder=1)

    # Draw nodes
    for key, (x, y, label, color, tc) in nodes.items():
        # Different widths for different label lengths
        if key in ("BFO", "GeoCore", "GeoReservoir"):
            w = 1.9
        elif key == "InterparticleP":
            w = 2.2
        else:
            w = 1.85
        ax.add_patch(FancyBboxPatch((x - w / 2, y - 0.25), w, 0.5,
                                      boxstyle="round,pad=0.02,rounding_size=0.08",
                                      facecolor=color, edgecolor=COL_TITLE, linewidth=1.0, zorder=2))
        ax.text(x, y, label,
                ha="center", va="center", fontsize=9.5,
                color=tc, fontweight="bold", zorder=3)

    # Legend
    legend_y = 0.3
    legend_items = [
        ("BFO (upper)",       COL_PALEBLUE),
        ("GeoCore (upper)",   COL_LIGHTBLUE),
        ("GeoReservoir (upper)", COL_BLUE),
        ("Pre-Salt term",     COL_ORANGE),
    ]
    lx = 1.0
    for label, color in legend_items:
        ax.add_patch(Rectangle((lx, legend_y), 0.3, 0.2, facecolor=color, edgecolor=COL_TITLE, linewidth=0.7))
        ax.text(lx + 0.4, legend_y + 0.1, label,
                ha="left", va="center", fontsize=9, color=COL_TEXT)
        lx += 2.7

    # Note in its own row above legend, right-aligned
    ax.text(14.5, 0.4,
            "Edges: rdfs:subClassOf",
            ha="right", fontsize=9, color=COL_GRAY, style="italic")

    _save(fig, "fig7_taxonomy_tree.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 9 — EXTENDED PIPELINE ARCHITECTURE (current work)
# ═════════════════════════════════════════════════════════════════════════════
def fig9_extended_pipeline():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(8, 8.5, "Ongoing Work — 7-Step Pipeline + RAG",
            ha="center", fontsize=18, fontweight="bold", color=COL_TITLE)
    ax.text(8, 8.05, "Blue = matches the paper   ·   Orange = new since the paper",
            ha="center", fontsize=10.5, color=COL_GRAY, style="italic")

    # Steps: (step_num, label, x, y, is_new, sub_label)
    steps = [
        (0,  "Ingest",          1.2, 6.4, False, "PDF → Markdown"),
        (None, "RAG",           1.2, 4.0, True,  "ChromaDB + BM25"),
        (1,  "Extract",         3.7, 6.4, False, "LLM term extraction"),
        (2,  "Aggregate",       6.0, 6.4, False, "Frequency counts"),
        (3,  "Filter",          8.3, 6.4, False, "≥7 papers"),
        (4,  "NLD",             10.6, 6.4, True, "RAG + Aristotelian"),
        (5,  "Categorize",      12.9, 6.4, False, "BFO/GeoCore/GeoR."),
        (6,  "Taxonomy",        12.9, 4.4, False, "Parent-child IS-A"),
        ("6b", "Relations",     10.6, 4.4, True,  "BFO/RO properties"),
        ("6c", "Critic",         8.3, 4.4, True,  "3-pass LLM review"),
        (7,  "OWL Export",       6.0, 4.4, True,  "Protégé .ttl"),
        ("7b", "Verify",         3.7, 4.4, True,  "OOPS + reasoner"),
    ]

    # Edges (from_label, to_label)
    edges = [
        ("Ingest", "Extract"),
        ("Ingest", "RAG"),
        ("Extract", "Aggregate"),
        ("Aggregate", "Filter"),
        ("Filter", "NLD"),
        ("NLD", "Categorize"),
        ("Categorize", "Taxonomy"),
        ("Taxonomy", "Relations"),
        ("Relations", "Critic"),
        ("Critic", "OWL Export"),
        ("OWL Export", "Verify"),
    ]
    # Dashed edges (retrieval context)
    dashed_edges = [
        ("RAG", "NLD"),
        ("RAG", "Categorize"),
    ]

    # Build lookup
    coords = {label: (x, y, is_new) for (_, label, x, y, is_new, _) in steps}

    # Draw edges
    def draw_edge(label_a, label_b, dashed=False):
        xa, ya, _ = coords[label_a]
        xb, yb, _ = coords[label_b]
        color = COL_ORANGE if dashed else "#7F7F7F"
        style = "dashed" if dashed else "solid"
        # Determine attach side
        if abs(xa - xb) < 0.1:
            # vertical
            ay = ya - 0.3 if ya > yb else ya + 0.3
            by = yb + 0.3 if yb < ya else yb - 0.3
            ax.annotate("", xy=(xb, by), xytext=(xa, ay),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.8,
                                        linestyle=style, shrinkA=0, shrinkB=0))
        else:
            ax_ = xa + 0.85 if xa < xb else xa - 0.85
            bx_ = xb - 0.85 if xb > xa else xb + 0.85
            ax.annotate("", xy=(bx_, yb), xytext=(ax_, ya),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.8,
                                        linestyle=style, shrinkA=0, shrinkB=0))

    for a, b in edges:
        draw_edge(a, b, dashed=False)
    for a, b in dashed_edges:
        draw_edge(a, b, dashed=True)

    # Draw nodes
    for step_num, label, x, y, is_new, sub in steps:
        color = COL_ORANGE if is_new else COL_BLUE
        # Box
        ax.add_patch(FancyBboxPatch((x - 0.85, y - 0.3), 1.7, 0.6,
                                      boxstyle="round,pad=0.02,rounding_size=0.08",
                                      facecolor=color, edgecolor=COL_TITLE, linewidth=1.2, zorder=2))
        ax.text(x, y, label,
                ha="center", va="center", fontsize=10.5, color="white", fontweight="bold", zorder=3)
        # Step number badge
        if step_num is not None:
            ax.add_patch(plt.Circle((x - 0.7, y + 0.3), 0.15,
                                     facecolor="white", edgecolor=color, linewidth=1.5, zorder=4))
            ax.text(x - 0.7, y + 0.3, str(step_num),
                    ha="center", va="center", fontsize=8, color=color, fontweight="bold", zorder=5)
        # Sub-label
        ax.text(x, y - 0.55, sub,
                ha="center", fontsize=8, color=COL_GRAY, style="italic")

    # Final output indicator
    ax.text(1.2, 2.9, "→  .ttl ontology",
            ha="center", fontsize=10, color=COL_GREEN, fontweight="bold")

    # Legend
    legend_y = 1.4
    ax.text(0.5, legend_y + 0.5, "Legend",
            fontsize=10, fontweight="bold", color=COL_TITLE)
    legend_items = [
        ("Step from the original paper",       COL_BLUE,    "-"),
        ("New since the paper",                COL_ORANGE,  "-"),
        ("RAG retrieval context",              COL_ORANGE,  "--"),
    ]
    ly = legend_y + 0.1
    for text, color, linestyle in legend_items:
        if linestyle == "-":
            ax.add_patch(Rectangle((0.5, ly - 0.07), 0.3, 0.14, facecolor=color, edgecolor="none"))
        else:
            ax.plot([0.5, 0.8], [ly, ly], color=color, linestyle="dashed", linewidth=2)
        ax.text(0.95, ly, text,
                fontsize=9.5, color=COL_TEXT, va="center")
        ly -= 0.32

    # Bottom summary
    ax.text(8, 0.5,
            "Paper: 4 phases (Corpus → Extraction → Classification → Validation)   →   "
            "Current: 7-step pipeline + 4-condition ablation (A/B/C/D)",
            ha="center", fontsize=10.5, color=COL_TITLE, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0F4FA",
                      edgecolor=COL_TITLE, linewidth=1.0))

    _save(fig, "fig9_extended_pipeline.png")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
ALL_FIGURES = {
    "1": ("Upper Ontology Stack", fig1_ontology_stack),
    "2": ("NLD Anatomy",          fig2_nld_anatomy),
    "3": ("Pipeline 4 Phases",    fig3_pipeline_4phases),
    "4": ("Four Pipelines",       fig4_four_pipelines),
    "5": ("Classification Waterfall", fig5_waterfall),
    "6": ("Expert Evaluation",    fig6_expert_eval),
    "7": ("Taxonomy Tree",        fig7_taxonomy_tree),
    "9": ("Extended Pipeline",    fig9_extended_pipeline),
}


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    if target == "all":
        for key, (name, func) in ALL_FIGURES.items():
            print(f"[fig {key}] {name}")
            func()
    elif target in ALL_FIGURES:
        name, func = ALL_FIGURES[target]
        print(f"[fig {target}] {name}")
        func()
    else:
        print(f"Unknown figure: {target}")
        print(f"Available: {', '.join(ALL_FIGURES)} or 'all'")
        sys.exit(1)


if __name__ == "__main__":
    main()
