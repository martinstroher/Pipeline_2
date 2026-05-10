"""
Build a new, text-reduced version of ICEIS2026_presentation_v2.pptx.
Keeps all shapes/formatting from the original, trims text, and adds charts.
"""
from copy import deepcopy
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.chart.data import CategoryChartData
from pptx.oxml.ns import qn
import os

SRC = "docs/ICEIS2026_presentation_v2.pptx"
DST = "docs/ICEIS2026_presentation_v6.pptx"

prs = Presentation(SRC)

# ── helpers ──────────────────────────────────────────────────────────────
def set_text(shape, text, bold=None, size=None, color=None):
    """Replace ALL text in a shape with a single paragraph."""
    tf = shape.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    if bold is not None:
        run.font.bold = bold
    if size is not None:
        run.font.size = size
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)

def set_bullets(shape, items, size=None, color=None, bold=None):
    """Replace text with multiple bullet-point paragraphs."""
    tf = shape.text_frame
    tf.clear()
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        run = p.add_run()
        run.text = item
        if size is not None:
            run.font.size = size
        if color is not None:
            run.font.color.rgb = RGBColor.from_string(color)
        if bold is not None:
            run.font.bold = bold

def find_shape(slide, name):
    """Find a shape by its name attribute."""
    for s in slide.shapes:
        if s.name == name:
            return s
    return None

# ── Gather slides ────────────────────────────────────────────────────────
slides = list(prs.slides)

# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 1 — Title (keep as-is, already clean)
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 2 — Context: The Knowledge Acquisition Bottleneck
#   4 blocks following the paper's narrative:
#   1. What are ontologies & why useful
#   2. Why slow to build
#   3. Our domain + upper ontologies + purpose
#   4. Our approach — accelerate, but taxonomy-level (not full ontology yet)
# ═══════════════════════════════════════════════════════════════════════════
s2 = slides[1]

# Title (TB1) — keep original
# Subtitle (TB2) — broaden to match 4-beat narrative
tb2_s2 = find_shape(s2, "TextBox 2")
if tb2_s2:
    set_text(tb2_s2, "From formal knowledge models to automated taxonomy learning",
             bold=False, size=Pt(14), color="555555")

# ── Block 1: What & Why ──
# Header (TB7)
tb7_s2 = find_shape(s2, "TextBox 7")
if tb7_s2:
    set_text(tb7_s2, "Ontologies: formal models of domain knowledge",
             bold=True, size=Pt(14), color="1F3864")

# Body (TB8)
tb8 = find_shape(s2, "TextBox 8")
if tb8:
    set_bullets(tb8, [
        "Shared vocabulary with formal semantics",
        "Enable reasoning, interoperability, consistency checking",
        "Backbone for data models & retrieval systems",
    ], size=Pt(13), color="1A1A2E")

# ── Block 2: Why slow ──
# Header (TB9)
tb9_s2 = find_shape(s2, "TextBox 9")
if tb9_s2:
    set_text(tb9_s2, "Why building them is slow",
             bold=True, size=Pt(14), color="1F3864")

# Body (TB10)
tb10 = find_shape(s2, "TextBox 10")
if tb10:
    set_bullets(tb10, [
        "Terminology acquisition + conceptualization (NeOn methodology)",
        "Requires dual expertise: domain + ontology engineering",
        "Cannot keep pace with exponential growth of literature",
    ], size=Pt(13), color="1A1A2E")

# ── Block 3: Our domain & purpose ──
# Header (TB12)
tb12_s2 = find_shape(s2, "TextBox 12")
if tb12_s2:
    set_text(tb12_s2, "Our context",
             bold=True, size=Pt(14), color="1F3864")

# Body (TB13)
tb13 = find_shape(s2, "TextBox 13")
if tb13:
    set_bullets(tb13, [
        "Domain: Brazilian Pre-Salt carbonate reservoirs",
        "Upper ontologies: BFO → GeoCore → GeoReservoir",
        "Goal: formal taxonomy of geological concepts",
    ], size=Pt(13), color="1A1A2E")

# ── Block 4: Our approach ──
# Header (TB15)
tb15_s2 = find_shape(s2, "TextBox 15")
if tb15_s2:
    set_text(tb15_s2, "Our approach",
             bold=True, size=Pt(14), color="1F3864")

# Body (TB16)
tb16 = find_shape(s2, "TextBox 16")
if tb16:
    set_text(tb16,
        "LLM-driven taxonomy learning — no fine-tuning, no labeled data\n"
        "Automates terminology + classification into upper-ontology hierarchies",
        bold=False, size=Pt(13), color="1A1A2E")

# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 3 — Research Question & Core Insight
#   PROBLEM: Long RQ text + NLD explanation is verbose
# ═══════════════════════════════════════════════════════════════════════════
s3 = slides[2]

# Shorten the research question
tb8 = find_shape(s3, "TextBox 8")
if tb8:
    set_text(tb8,
        'Can a general-purpose LLM extract terms and classify them into formal '
        'upper-ontology hierarchies — without fine-tuning?',
        bold=True, size=Pt(17), color="1F3864")

# NLD explanation — keywords
tb12 = find_shape(s3, "TextBox 12")
if tb12:
    set_text(tb12,
        '"X  is a  Y  that  Z"\n'
        'genus + differentia → human & LLM parseable',
        bold=False, size=Pt(13), color="1A1A2E")

# Lopes Junior insight — keyword versions
tb20 = find_shape(s3, "TextBox 20")
if tb20:
    set_text(tb20,
        ">90% macro-F1 for BFO classification",
        bold=False, size=Pt(13), color="1A1A2E")

tb23 = find_shape(s3, "TextBox 23")
if tb23:
    set_text(tb23,
        "Best representation — cross-language",
        bold=False, size=Pt(13), color="1A1A2E")

tb26 = find_shape(s3, "TextBox 26")
if tb26:
    set_text(tb26,
        "First: NLDs for extraction + hierarchy mapping",
        bold=False, size=Pt(13), color="1A1A2E")

# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 4 — Methodology Overview (4 phases)
#   Already quite visual (4 cards), but card descriptions can be trimmed
# ═══════════════════════════════════════════════════════════════════════════
s4 = slides[3]

# Phase 1 description
tb11 = find_shape(s4, "TextBox 11")
if tb11:
    set_bullets(tb11, [
        "40 peer-reviewed Pre-Salt articles",
        "Senior specialist curated",
    ], size=Pt(12.5), color="1A1A2E")

# Phase 2 description
tb17 = find_shape(s4, "TextBox 17")
if tb17:
    set_bullets(tb17, [
        "LLM + Corpus (extractive)",
        "LLM Generated (zero-shot)",
        "NER + Corpus (hybrid)",
        "TF-IDF (baseline)",
    ], size=Pt(12.5), color="1A1A2E")

# Phase 3 description
tb23 = find_shape(s4, "TextBox 23")
if tb23:
    set_bullets(tb23, [
        "Aristotelian NLD generation",
        "CoT categorization",
        "BFO → GeoCore → GeoReservoir",
    ], size=Pt(12.5), color="1A1A2E")

# Phase 4 description
tb29 = find_shape(s4, "TextBox 29")
if tb29:
    set_bullets(tb29, [
        "5 domain experts (PhD + MSc)",
        "Relevance, NLD, category eval",
    ], size=Pt(12.5), color="1A1A2E")

# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 5 — Four Term Extraction Pipelines
#   PROBLEM: Each pipeline card has 2-3 sentences → trim to 1-2 lines
# ═══════════════════════════════════════════════════════════════════════════
s5 = slides[4]

# P1 — LLM + Corpus
tb7 = find_shape(s5, "TextBox 6")
if tb7:
    set_text(tb7,
        "LLM extracts terms from each paper\n"
        "Top 100 by frequency",
        bold=False, size=Pt(13), color="1A1A2E")

# P2 — LLM Generated
tb11 = find_shape(s5, "TextBox 10")
if tb11:
    set_text(tb11,
        "Terms from LLM knowledge only\n"
        "No corpus → tests recall",
        bold=False, size=Pt(13), color="1A1A2E")

# P3 — NER + Corpus
tb15 = find_shape(s5, "TextBox 14")
if tb15:
    set_text(tb15,
        "spaCy NER on corpus\n"
        "NER vs. general-purpose LLM",
        bold=False, size=Pt(13), color="1A1A2E")

# P4 — TF-IDF
tb21 = find_shape(s5, "TextBox 21")
if tb21:
    set_text(tb21,
        "TF-IDF on 40-article corpus\n"
        "Traditional frequency baseline",
        bold=False, size=Pt(13), color="1A1A2E")

# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 6 — NLD-Centric Classification Workflow
#   Three step cards + three example cards. Trim step descriptions.
# ═══════════════════════════════════════════════════════════════════════════
s6 = slides[5]

# Step 1 — NLD Generation (TextBox 10)
tb10_s6 = find_shape(s6, "TextBox 10")
if tb10_s6:
    set_text(tb10_s6,
        'Persona: "Senior geoscientist + ontology engineer"\n'
        '"X is a Y that Z"',
        bold=False, size=Pt(12.5), color="1A1A2E")

# Step 2 — CoT Categorization (TextBox 15)
tb15_s6 = find_shape(s6, "TextBox 15")
if tb15_s6:
    set_text(tb15_s6,
        "CoT: term + NLD → category\n"
        "Waterfall: GeoReservoir → GeoCore → BFO",
        bold=False, size=Pt(12.5), color="1A1A2E")

# Step 3 — Resulting Artifact (TextBox 20)
tb20_s6 = find_shape(s6, "TextBox 20")
if tb20_s6:
    set_text(tb20_s6,
        "BFO → GeoCore → GeoReservoir\n"
        "Expert-validated",
        bold=False, size=Pt(12.5), color="1A1A2E")

# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 7 — Expert Validation Design
#   Trim expert description and sampling strategy text
# ═══════════════════════════════════════════════════════════════════════════
s7 = slides[6]

# Expert description
tb9_s7 = find_shape(s7, "TextBox 9")
if tb9_s7:
    set_bullets(tb9_s7, [
        "PhD & MSc, up to 20 yrs",
        "Industry + Academia (Pre-Salt)",
        "100 reference terms (20/expert)",
    ], size=Pt(13), color="1A1A2E")

# Sampling strategy
tb13_s7 = find_shape(s7, "TextBox 13")
if tb13_s7:
    set_bullets(tb13_s7, [
        "100 terms/pipeline × 4 = 400 entries",
        "Anonymized, randomized",
        "Blind evaluation",
    ], size=Pt(13), color="1A1A2E")

# Trim evaluation criteria cards
# Term Relevance
tb18_s7 = find_shape(s7, "TextBox 18")
if tb18_s7:
    set_text(tb18_s7,
        "Relevant / Irrelevant / Invalid / Unknown",
        bold=False, size=Pt(12.5), color="1A1A2E")

# NLD Accuracy
tb23_s7 = find_shape(s7, "TextBox 23")
if tb23_s7:
    set_text(tb23_s7,
        "Correct / Partially Correct / Incorrect\nAmbiguous / Wrong Context / Unknown",
        bold=False, size=Pt(12.5), color="1A1A2E")

# Category Accuracy
tb28_s7 = find_shape(s7, "TextBox 28")
if tb28_s7:
    set_text(tb28_s7,
        "Correct & Specific / Correct but Unspecific\nIncorrect / Unknown",
        bold=False, size=Pt(12.5), color="1A1A2E")

# ═══════════════════════════════════════════════════════════════════════════
# SLIDES 8-9 — Results tables
#   These are data-heavy by nature. Keep tables as-is.
#   Only trim the commentary text below tables.
# ═══════════════════════════════════════════════════════════════════════════

# Slide 8: trim the 4 insight callout boxes on the right
s8 = slides[7]
tb58_s8 = find_shape(s8, "TextBox 58")
if tb58_s8:
    set_text(tb58_s8, "Best recall + precision",
             bold=False, size=Pt(13), color="1A1A2E")
tb61_s8 = find_shape(s8, "TextBox 61")
if tb61_s8:
    set_text(tb61_s8, "Corpus grounds the LLM",
             bold=False, size=Pt(13), color="1A1A2E")
tb64_s8 = find_shape(s8, "TextBox 64")
if tb64_s8:
    set_text(tb64_s8, "46% noise — no semantics",
             bold=False, size=Pt(13), color="1A1A2E")
tb67_s8 = find_shape(s8, "TextBox 67")
if tb67_s8:
    set_text(tb67_s8, "Decent recall, 33 disputed",
             bold=False, size=Pt(13), color="1A1A2E")

# Slide 9: trim analysis text
s9 = slides[8]
tb57_s9 = find_shape(s9, "TextBox 57")
if tb57_s9:
    set_text(tb57_s9,
        "LLM-Gen best NLDs — canonical terms",
        bold=False, size=Pt(11.5), color="555555")

tb111_s9 = find_shape(s9, "TextBox 111")
if tb111_s9:
    set_text(tb111_s9,
        "LLM+Corpus: 71% correct, 18% rejected",
        bold=False, size=Pt(12), color="1F3864")

# ── Slide 9: Color-code the categorization table cells ────────────────
def recolor_shape(slide, name, hex_color):
    """Recolor all text runs in a shape to the given hex color."""
    shape = find_shape(slide, name)
    if shape and shape.has_text_frame:
        for para in shape.text_frame.paragraphs:
            for run in para.runs:
                run.font.color.rgb = RGBColor.from_string(hex_color)

# Row: Majority Correct → blue
for tb_name in ["TextBox 71", "TextBox 72", "TextBox 74", "TextBox 76", "TextBox 78"]:
    recolor_shape(s9, tb_name, "2E75B6")

# Row: Majority Partial → gold
for tb_name in ["TextBox 81", "TextBox 82", "TextBox 84", "TextBox 86", "TextBox 88"]:
    recolor_shape(s9, tb_name, "DAA520")

# Row: Disputed/Rejected → red
for tb_name in ["TextBox 91", "TextBox 92", "TextBox 94", "TextBox 96", "TextBox 98"]:
    recolor_shape(s9, tb_name, "C0392B")

# Row: Strict Consensus → green
for tb_name in ["TextBox 101", "TextBox 102", "TextBox 104", "TextBox 106", "TextBox 108"]:
    recolor_shape(s9, tb_name, "1E8855")

# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 10 — Resulting Conceptual Artifact
#   Trim "What was produced" bullet list
# ═══════════════════════════════════════════════════════════════════════════
s10 = slides[9]
tb8_s10 = find_shape(s10, "TextBox 8")
if tb8_s10:
    set_bullets(tb8_s10, [
        "Unanimous expert consensus",
        "BFO → GeoCore → GeoReservoir hierarchy",
        "Classes only — no axioms yet",
        "Foundation for similarity-search",
    ], size=Pt(13.5), color="1A1A2E")

# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 11 — Open Problems
#   PROBLEM: Three cards each with 4-5 bullet lines → trim dramatically
# ═══════════════════════════════════════════════════════════════════════════
s11 = slides[10]

# Context Blindness — subtitle
tb11_s11 = find_shape(s11, "TextBox 11")
if tb11_s11:
    set_text(tb11_s11,
        "No source context → generic definitions",
        bold=False, size=Pt(12), color="1A1A2E")

# Context Blindness — bullets
tb12_s11 = find_shape(s11, "TextBox 12")
if tb12_s11:
    set_bullets(tb12_s11, [
        "34% NLDs only 'Partially Correct'",
        "Outdated paradigms propagated",
        "→ Fix: RAG at NLD generation time",
    ], size=Pt(11.5), color="1A1A2E")

# Ambiguity — subtitle
tb17_s11 = find_shape(s11, "TextBox 17")
if tb17_s11:
    set_text(tb17_s11,
        "Expert disagreement → categorization limits",
        bold=False, size=Pt(12), color="1A1A2E")

# Ambiguity — bullets
tb18_s11 = find_shape(s11, "TextBox 18")
if tb18_s11:
    set_bullets(tb18_s11, [
        "39 Strict vs 32 Majority-only",
        "LLM reasoning limited without grounding",
        "→ Fix: domain context to elevate consensus",
    ], size=Pt(11.5), color="1A1A2E")

# Granularity Gap — REMOVED (not grounded in the paper)
# Blank out the card shapes so they don't show text
tb23_s11 = find_shape(s11, "TextBox 23")
if tb23_s11:
    set_text(tb23_s11, "", size=Pt(1))

tb24_s11 = find_shape(s11, "TextBox 24")
if tb24_s11:
    set_text(tb24_s11, "", size=Pt(1))

# Also blank the card title shape (search by text content)
for shape in s11.shapes:
    if shape.has_text_frame:
        full_text = shape.text_frame.text.strip()
        if "Granularity" in full_text or "granularity" in full_text:
            set_text(shape, "", size=Pt(1))

# Hide the blue card background behind Granularity Gap
# It's the rightmost card rectangle — find by position (left > 7500000 EMU)
for shape in s11.shapes:
    if not shape.has_text_frame and shape.left is not None and shape.left > 7500000:
        # Make it transparent
        shape.fill.background()
        if hasattr(shape, 'line'):
            shape.line.fill.background()

# Bottom summary — updated for 2 problems only
tb27_s11 = find_shape(s11, "TextBox 27")
if tb27_s11:
    set_text(tb27_s11,
        "Both problems → same fix: RAG at NLD generation",
        bold=True, size=Pt(13.5), color="1F3864")

# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 12 — Ongoing Work & Conclusion
#   Trim the "Current Implementation" bullet list
# ═══════════════════════════════════════════════════════════════════════════
s12 = slides[11]

# Current implementation bullets
tb7_s12 = find_shape(s12, "TextBox 7")
if tb7_s12:
    set_bullets(tb7_s12, [
        "Hybrid RAG: BGE-M3 + BM25 + reranker",
        "NLDs grounded in source text",
        "80 papers (2× this study)",
        "4-condition ablation study",
        "Statistical expert eval: Wilcoxon, Friedman, ICC",
        "OWL/Turtle export (Protégé-compatible)",
    ], size=Pt(13.5), color="1A1A2E")

# ═══════════════════════════════════════════════════════════════════════════
# CHARTS — Add visual data representations to results slides
# ═══════════════════════════════════════════════════════════════════════════

COLORS_4 = ["2E75B6", "5B9BD5", "ED7D31", "C0392B"]  # blue, lightblue, orange, red

def style_chart(chart, font_size=Pt(10)):
    """Apply consistent styling to chart."""
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.BOTTOM
    chart.legend.include_in_layout = False
    chart.legend.font.size = font_size
    # Style value axis
    val_ax = chart.value_axis
    val_ax.has_title = False
    val_ax.tick_labels.font.size = font_size
    val_ax.major_gridlines.format.line.color.rgb = RGBColor(0xDD, 0xDD, 0xDD)
    # Style category axis
    cat_ax = chart.category_axis
    cat_ax.tick_labels.font.size = font_size

def color_series(plot, colors):
    """Apply colors to each series in a plot."""
    for i, series in enumerate(plot.series):
        if i < len(colors):
            fill = series.format.fill
            fill.solid()
            fill.fore_color.rgb = RGBColor.from_string(colors[i])

# ── Chart 1: Recall + Strictly Relevant (Slide 8) ─────────────────────
# Add a grouped bar chart in the bottom-right area of slide 8
# Data from the table already on the slide
chart_data_s8 = CategoryChartData()
chart_data_s8.categories = ["LLM+Corpus", "LLM Generated", "NER+Corpus", "TF-IDF"]
chart_data_s8.add_series("Recall (%)", (60, 40, 32, 25))
chart_data_s8.add_series("Relevant (%)", (97.8, 96.2, 86.6, 46.0))
chart_data_s8.add_series("Strict Consensus", (91, 88, 66, 25))

# Place below the table + callout area
chart_frame_s8 = s8.shapes.add_chart(
    XL_CHART_TYPE.COLUMN_CLUSTERED,
    Emu(411480), Emu(3700000),   # left, top
    Emu(8800000), Emu(2600000),  # width, height
    chart_data_s8
)
chart_s8 = chart_frame_s8.chart
style_chart(chart_s8, Pt(9))
chart_s8.value_axis.maximum_scale = 100
color_series(chart_s8.plots[0], ["2E75B6", "1E8855", "ED7D31"])

# Add data labels
for series in chart_s8.plots[0].series:
    series.has_data_labels = True
    series.data_labels.font.size = Pt(8)
    series.data_labels.number_format = '0'
    series.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END

# ── Chart 2: NLD Accuracy stacked bar (Slide 9) ──────────────────────
# Data from the NLD Accuracy table on slide 9
chart_data_nld = CategoryChartData()
chart_data_nld.categories = ["LLM+Corpus", "LLM Gen.", "NER+Corpus", "TF-IDF"]
chart_data_nld.add_series("Correct", (53.7, 66.0, 57.0, 34.7))
chart_data_nld.add_series("Partially Correct", (34.0, 24.3, 22.3, 22.0))
chart_data_nld.add_series("Incorrect", (9.7, 6.7, 15.3, 9.0))
chart_data_nld.add_series("Ambiguous/Unknown", (2.0, 2.6, 4.0, 30.4))

chart_frame_nld = s9.shapes.add_chart(
    XL_CHART_TYPE.COLUMN_STACKED,
    Emu(411480), Emu(4100000),   # left, top
    Emu(5200000), Emu(2300000),  # width, height
    chart_data_nld
)
chart_nld = chart_frame_nld.chart
style_chart(chart_nld, Pt(8))
chart_nld.value_axis.maximum_scale = 100
color_series(chart_nld.plots[0], ["1E8855", "5B9BD5", "C0392B", "999999"])

# ── Chart 3: Categorization accuracy grouped bar (Slide 9) ────────────
# 4 series: Strict Consensus, Majority Correct, Majority Partial, Disputed
chart_data_cat = CategoryChartData()
chart_data_cat.categories = ["LLM+Corpus", "LLM Gen.", "NER+Corpus", "TF-IDF"]
chart_data_cat.add_series("Strict Consensus", (39, 43, 23, 17))
chart_data_cat.add_series("Majority Correct", (71, 66, 68, 36))
chart_data_cat.add_series("Majority Partial", (11, 9, 10, 5))
chart_data_cat.add_series("Disputed/Rejected", (18, 25, 22, 59))

chart_frame_cat = s9.shapes.add_chart(
    XL_CHART_TYPE.COLUMN_CLUSTERED,
    Emu(5900000), Emu(4100000),  # left, top
    Emu(5900000), Emu(2300000),  # width, height
    chart_data_cat
)
chart_cat = chart_frame_cat.chart
style_chart(chart_cat, Pt(8))
chart_cat.value_axis.maximum_scale = 100
# Colors: green=Strict, blue=Majority, gold=Partial, red=Disputed
color_series(chart_cat.plots[0], ["1E8855", "2E75B6", "DAA520", "C0392B"])

# Add data labels to categorization chart
for series in chart_cat.plots[0].series:
    series.has_data_labels = True
    series.data_labels.font.size = Pt(8)
    series.data_labels.number_format = '0'
    series.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END

# ── Save ─────────────────────────────────────────────────────────────────
prs.save(DST)
print(f"Saved to {DST}")
print(f"Size: {os.path.getsize(DST):,} bytes")
