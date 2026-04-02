"""
Generate ICEIS 2026 presentation for:
LLM-Driven Ontology Learning: From Term Extraction to Upper-Level
Categorization for Building Information System Models
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from lxml import etree

# ── Palette ─────────────────────────────────────────────────────────────────
DARK_BLUE  = RGBColor(0x1F, 0x38, 0x64)
MID_BLUE   = RGBColor(0x2E, 0x75, 0xB6)
PALE_BLUE  = RGBColor(0xD6, 0xE4, 0xF7)
ORANGE     = RGBColor(0xED, 0x7D, 0x31)
WHITE      = RGBColor(0xFF, 0xFF, 0xFF)
NEAR_BLACK = RGBColor(0x1A, 0x1A, 0x2E)
MED_GRAY   = RGBColor(0x55, 0x55, 0x55)
LIGHT_GRAY = RGBColor(0xF4, 0xF6, 0xF9)
GREEN      = RGBColor(0x1E, 0x88, 0x55)
AMBER      = RGBColor(0xF5, 0xA6, 0x23)
RED        = RGBColor(0xC0, 0x39, 0x2B)

# ── Slide dimensions (widescreen 16:9) ──────────────────────────────────────
W = Inches(13.33)
H = Inches(7.5)
HEADER_H  = Inches(1.15)
ACCENT_H  = Inches(0.06)
MARGIN_X  = Inches(0.45)
CONTENT_Y = HEADER_H + ACCENT_H + Inches(0.15)
CONTENT_H = H - CONTENT_Y - Inches(0.2)


# ── Low-level helpers ────────────────────────────────────────────────────────
def _blank_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def add_rect(slide, x, y, w, h, fill, line=None, alpha=None):
    from pptx.enum.dml import MSO_THEME_COLOR
    sp = slide.shapes.add_shape(1, x, y, w, h)
    sp.fill.solid()
    sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line
        sp.line.width = Pt(0.5)
    return sp


def add_text(slide, x, y, w, h, text, size, bold=False, italic=False,
             color=NEAR_BLACK, align=PP_ALIGN.LEFT, wrap=True, spacing_after=0):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    if spacing_after:
        p.space_after = Pt(spacing_after)
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = 'Calibri'
    return tb


def add_bullets(slide, x, y, w, h, items, size=15, color=NEAR_BLACK,
                indent_size=14, bold_first=False, line_spacing=1.15):
    """items: list of (level, text) tuples. level 0 = top, 1 = sub."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    for level, text in items:
        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()
        p.level = level
        p.space_before = Pt(2 if level == 0 else 1)
        p.space_after = Pt(1)
        # indent
        pPr = p._pPr
        if pPr is None:
            pPr = p._p.get_or_add_pPr()
        # bullet
        bullet_xml = f'<a:buChar xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" char="{"•" if level==0 else "–"}"/>'
        buChar = etree.fromstring(bullet_xml)
        # remove existing buNone/buChar
        for child in list(pPr):
            if child.tag.endswith(('buNone', 'buChar', 'buFont', 'buClr', 'buSzPct')):
                pPr.remove(child)
        pPr.append(buChar)
        indent_val = Inches(0.25 * (level + 1))
        marL = int(indent_val)
        indent = -int(Inches(0.2))
        pPr.set('marL', str(marL))
        pPr.set('indent', str(indent))
        run = p.add_run()
        run.text = text
        run.font.size = Pt(size - level * 1.5)
        run.font.bold = bold_first and level == 0
        run.font.color.rgb = color
        run.font.name = 'Calibri'
    return tb


# ── Slide template header ────────────────────────────────────────────────────
def add_header(slide, title, subtitle=None):
    # Background header bar
    add_rect(slide, 0, 0, W, HEADER_H, DARK_BLUE)
    # Orange accent line
    add_rect(slide, 0, HEADER_H, W, ACCENT_H, ORANGE)
    # Title
    add_text(slide, MARGIN_X, Inches(0.12), W - MARGIN_X * 2, Inches(0.75),
             title, size=26, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
    if subtitle:
        add_text(slide, MARGIN_X, Inches(0.82), W - MARGIN_X * 2, Inches(0.3),
                 subtitle, size=14, italic=True, color=PALE_BLUE, align=PP_ALIGN.LEFT)


def add_footer(slide, text="ICEIS 2026"):
    add_text(slide, Inches(0.3), H - Inches(0.35), W - Inches(0.6), Inches(0.3),
             text, size=9, color=MED_GRAY, align=PP_ALIGN.RIGHT)


# ── SLIDE 1 — Title ──────────────────────────────────────────────────────────
def slide_title(prs):
    slide = _blank_slide(prs)
    # Full dark-blue background
    add_rect(slide, 0, 0, W, H, DARK_BLUE)
    # Orange stripe (decorative)
    add_rect(slide, 0, Inches(5.05), W, Inches(0.07), ORANGE)
    # Pale blue bottom strip
    add_rect(slide, 0, Inches(5.12), W, H - Inches(5.12), RGBColor(0x16, 0x22, 0x40))

    # Title
    add_text(slide, MARGIN_X, Inches(1.2), W - MARGIN_X * 2, Inches(1.5),
             "LLM-Driven Ontology Learning", size=40, bold=True, color=WHITE,
             align=PP_ALIGN.CENTER)
    add_text(slide, MARGIN_X, Inches(2.65), W - MARGIN_X * 2, Inches(0.7),
             "From Term Extraction to Upper-Level Categorization", size=26,
             color=PALE_BLUE, align=PP_ALIGN.CENTER)
    add_text(slide, MARGIN_X, Inches(3.25), W - MARGIN_X * 2, Inches(0.55),
             "for Building Information System Models", size=22,
             color=PALE_BLUE, align=PP_ALIGN.CENTER)

    # Authors
    add_text(slide, MARGIN_X, Inches(4.2), W - MARGIN_X * 2, Inches(0.5),
             "Martin Ströher  ·  Thaís Schäfer Luiz  ·  Eduardo Roemers-Oliveira  ·  Lucas Valadares Vieira",
             size=14, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(slide, MARGIN_X, Inches(4.65), W - MARGIN_X * 2, Inches(0.45),
             "Fábio Herbert Jones  ·  Luiz Fernando De Ros  ·  Mara Abel",
             size=14, color=WHITE, align=PP_ALIGN.CENTER)

    # Institutions
    add_text(slide, MARGIN_X, Inches(5.25), W - MARGIN_X * 2, Inches(0.45),
             "UFRGS  ·  UFRJ  ·  Petrobras Research Center  ·  Colorado State University",
             size=13, color=PALE_BLUE, align=PP_ALIGN.CENTER)

    # Conference
    add_text(slide, MARGIN_X, Inches(6.6), W - MARGIN_X * 2, Inches(0.5),
             "ICEIS 2026  —  International Conference on Enterprise Information Systems",
             size=13, bold=True, color=AMBER, align=PP_ALIGN.CENTER)


# ── SLIDE 2 — Context: The Bottleneck ────────────────────────────────────────
def slide_context(prs):
    slide = _blank_slide(prs)
    add_header(slide, "Context: The Knowledge Acquisition Bottleneck",
               "Why building domain ontologies remains painfully slow")
    add_footer(slide)

    # Left column: problem description
    LW = Inches(6.8)
    add_text(slide, MARGIN_X, CONTENT_Y, LW, Inches(0.45),
             "Ontologies power Knowledge-Intensive Information Systems", size=16,
             bold=True, color=DARK_BLUE)

    bullets_left = [
        (0, "Formal backbone for data models, software applications, retrieval systems"),
        (0, "Enterprise ontology projects rely on specialized vocabulary"),
        (0, "Vocabulary must be mapped to formal upper-level hierarchies"),
        (1, "BFO  →  GeoCore  →  Domain ontology"),
    ]
    add_bullets(slide, MARGIN_X, CONTENT_Y + Inches(0.55), LW, Inches(1.5),
                bullets_left, size=15)

    add_text(slide, MARGIN_X, CONTENT_Y + Inches(2.1), LW, Inches(0.45),
             "NeOn Methodology identifies two bottleneck steps:", size=16,
             bold=True, color=DARK_BLUE)
    bullets_bottleneck = [
        (0, "Non-Ontological Resource Reuse  — selecting terminology sources"),
        (0, "Conceptualization  — clarifying and formalizing semantics"),
        (1, "Both require highly specialized experts for extended sessions"),
        (1, "Cannot keep pace with the exponential growth of unstructured data"),
    ]
    add_bullets(slide, MARGIN_X, CONTENT_Y + Inches(2.55), LW, Inches(1.8),
                bullets_bottleneck, size=15)

    # Right column: existing solutions + gap
    RX = MARGIN_X + LW + Inches(0.3)
    RW = W - RX - MARGIN_X
    add_rect(slide, RX - Inches(0.1), CONTENT_Y - Inches(0.1),
             RW + Inches(0.2), Inches(4.8), LIGHT_GRAY)

    add_text(slide, RX, CONTENT_Y + Inches(0.1), RW, Inches(0.4),
             "Existing approaches", size=16, bold=True, color=DARK_BLUE)

    existing = [
        (0, "Statistical (TF-IDF): fast but semantically weak"),
        (0, "Embedding models (BERT): better, but require labeled data + fine-tuning"),
        (0, "Fine-tuned LLMs: shifts the bottleneck, not removes it"),
    ]
    add_bullets(slide, RX, CONTENT_Y + Inches(0.55), RW, Inches(1.6),
                existing, size=14)

    add_rect(slide, RX, CONTENT_Y + Inches(2.25), RW, Inches(0.05), ORANGE)
    add_text(slide, RX, CONTENT_Y + Inches(2.4), RW, Inches(0.45),
             "Our path:", size=15, bold=True, color=MID_BLUE)
    add_text(slide, RX, CONTENT_Y + Inches(2.85), RW, Inches(1.6),
             "Apply a general-purpose LLM without any domain-specific fine-tuning"
             " — more accessible, more generalizable.",
             size=14, color=NEAR_BLACK, wrap=True)


# ── SLIDE 3 — Research Question & Key Insight ────────────────────────────────
def slide_rq(prs):
    slide = _blank_slide(prs)
    add_header(slide, "Research Question & Core Insight")
    add_footer(slide)

    # Big RQ box
    add_rect(slide, MARGIN_X, CONTENT_Y, W - MARGIN_X * 2, Inches(1.35), PALE_BLUE)
    add_rect(slide, MARGIN_X, CONTENT_Y, Inches(0.08), Inches(1.35), MID_BLUE)
    add_text(slide, MARGIN_X + Inches(0.2), CONTENT_Y + Inches(0.12),
             W - MARGIN_X * 2 - Inches(0.4), Inches(1.1),
             "Can a non-specialist LLM produce semantically precise ontological artifacts "
             "— term extraction and taxonomic categorization into formal hierarchies — "
             "to bootstrap ontology construction, without fine-tuning?",
             size=18, bold=True, color=DARK_BLUE, align=PP_ALIGN.LEFT)

    # ── What is an NLD? ──
    NLD_Y = CONTENT_Y + Inches(1.55)
    add_text(slide, MARGIN_X, NLD_Y, W - MARGIN_X * 2, Inches(0.38),
             "What is a Natural Language Definition (NLD)?", size=16, bold=True, color=DARK_BLUE)

    # Definition + example side by side
    DEF_W = Inches(5.5)
    EX_X  = MARGIN_X + DEF_W + Inches(0.35)
    EX_W  = W - EX_X - MARGIN_X
    BOX_H = Inches(1.55)
    BY    = NLD_Y + Inches(0.42)

    # Definition box (left)
    add_rect(slide, MARGIN_X, BY, DEF_W, BOX_H, LIGHT_GRAY)
    add_rect(slide, MARGIN_X, BY, Inches(0.07), BOX_H, MID_BLUE)
    add_text(slide, MARGIN_X + Inches(0.17), BY + Inches(0.1),
             DEF_W - Inches(0.28), Inches(1.35),
             "A concise, Aristotelian-style definition that clarifies a term's meaning:\n\n"
             "\"X  is a  Y  that  Z\"\n\n"
             "genus (Y) + differentia (Z) — easy to parse for both humans and LLMs",
             size=13.5, color=NEAR_BLACK, wrap=True)

    # Example box (right)
    add_rect(slide, EX_X, BY, EX_W, BOX_H, PALE_BLUE)
    add_rect(slide, EX_X, BY, Inches(0.07), BOX_H, ORANGE)
    add_text(slide, EX_X + Inches(0.17), BY + Inches(0.08),
             EX_W - Inches(0.28), Inches(0.3),
             "Term:  Coquina", size=13, bold=True, color=DARK_BLUE)
    add_text(slide, EX_X + Inches(0.17), BY + Inches(0.38),
             EX_W - Inches(0.28), Inches(1.1),
             "\"Coquina is a sedimentary rock that is composed of cemented bioclastic "
             "fragments — mainly mollusk shells — deposited in high-energy shallow "
             "marine environments, and is a key Pre-Salt reservoir rock in Brazil.\"",
             size=12, italic=True, color=NEAR_BLACK, wrap=True)

    # ── Key Insight ──
    IY = BY + BOX_H + Inches(0.22)
    add_text(slide, MARGIN_X, IY, W - MARGIN_X * 2, Inches(0.35),
             "Key Insight — Lopes Junior (2025)", size=16, bold=True, color=DARK_BLUE)

    cols = [
        (Inches(0.45), Inches(3.85),
         "Language models using NLDs to classify domain entities into top-level "
         "ontology concepts achieve >90% macro F1-score",
         MID_BLUE),
        (Inches(4.7), Inches(3.85),
         "NLDs are the most effective textual representation for ontological classification, "
         "even across multilingual settings",
         MID_BLUE),
        (Inches(9.0), Inches(3.85),
         "No prior work had leveraged this NLD insight for corpus-grounded extraction "
         "and formal hierarchy mapping",
         ORANGE),
    ]
    for x, w, txt, clr in cols:
        add_rect(slide, x, IY + Inches(0.42), w, Inches(1.75), LIGHT_GRAY)
        add_rect(slide, x, IY + Inches(0.42), w, Inches(0.06), clr)
        add_text(slide, x + Inches(0.15), IY + Inches(0.57), w - Inches(0.3),
                 Inches(1.5), txt, size=13.5, color=NEAR_BLACK, wrap=True)


# ── SLIDE 4 — Methodology Overview ───────────────────────────────────────────
def slide_method_overview(prs):
    slide = _blank_slide(prs)
    add_header(slide, "Methodology Overview", "4 phases — applied to 40 curated Brazilian Pre-Salt scientific articles")
    add_footer(slide)

    phases = [
        ("Phase 1", "Corpus\nPreparation",
         "40 peer-reviewed Pre-Salt articles\nCurated by senior domain specialist\nAutomatic text extraction (pdftotext)"),
        ("Phase 2", "Term Acquisition\n(4 Pipelines)",
         "LLM + Corpus (extractive)\nLLM Generated (zero-shot)\nNER + Corpus (hybrid)\nTF-IDF (baseline)"),
        ("Phase 3", "NLD-Centric\nClassification",
         "LLM generates Aristotelian NLD\nChain-of-Thought categorization\nBFO → GeoCore → GeoReservoir"),
        ("Phase 4", "Specialist\nValidation",
         "5 domain experts (PhD + MSc)\nTerm Relevance, NLD Accuracy,\nCategory Accuracy")
    ]

    PW = Inches(2.8)
    PH = Inches(4.2)
    gap = Inches(0.28)
    total = 4 * PW + 3 * gap
    start_x = (W - total) / 2

    for i, (phase, title, detail) in enumerate(phases):
        px = start_x + i * (PW + gap)
        py = CONTENT_Y + Inches(0.1)

        add_rect(slide, px, py, PW, PH, PALE_BLUE)
        add_rect(slide, px, py, PW, Inches(0.06), MID_BLUE if i < 3 else ORANGE)

        # Phase label
        add_text(slide, px, py + Inches(0.1), PW, Inches(0.35),
                 phase, size=12, bold=True, color=MID_BLUE, align=PP_ALIGN.CENTER)

        # Phase title
        add_text(slide, px + Inches(0.1), py + Inches(0.45), PW - Inches(0.2), Inches(0.9),
                 title, size=15, bold=True, color=DARK_BLUE, align=PP_ALIGN.CENTER)

        # Details
        add_text(slide, px + Inches(0.12), py + Inches(1.4), PW - Inches(0.24), Inches(2.7),
                 detail, size=13, color=NEAR_BLACK, wrap=True)

        # Arrow between boxes
        if i < 3:
            ax = px + PW + Inches(0.03)
            ay = py + PH / 2 - Inches(0.15)
            add_text(slide, ax, ay, gap - Inches(0.06), Inches(0.3),
                     "→", size=22, bold=True, color=MID_BLUE, align=PP_ALIGN.CENTER)

    # Model info strip
    add_rect(slide, MARGIN_X, H - Inches(0.75), W - MARGIN_X * 2, Inches(0.42), PALE_BLUE)
    add_text(slide, MARGIN_X + Inches(0.1), H - Inches(0.7), W - MARGIN_X * 2 - Inches(0.2), Inches(0.38),
             "Model: Gemini 2.5 Pro  ·  Temperature = 0 (deterministic)  ·  "
             "Ontology stack: BFO → GeoCore → GeoReservoir",
             size=13, color=DARK_BLUE, align=PP_ALIGN.CENTER)


# ── SLIDE 5 — Four Extraction Pipelines ──────────────────────────────────────
def slide_pipelines(prs):
    slide = _blank_slide(prs)
    add_header(slide, "Phase 2: The Four Term Extraction Pipelines")
    add_footer(slide)

    pipelines = [
        ("P1 — LLM + Corpus", MID_BLUE,
         "Extractive mode: LLM reads each of the 40 articles and extracts "
         "all relevant geological terms directly from the text.\n\n"
         "Stemming + normalization applied for aggregation."),
        ("P2 — LLM Generated", ORANGE,
         "Zero-shot, corpus-agnostic: LLM generates the 500 most important "
         "domain terms from its pre-trained knowledge alone.\n\n"
         "Temperature = 0.2 for slight lexical variance."),
        ("P3 — NER + Corpus", RGBColor(0x5B, 0x2C, 0x8D),
         "Hybrid / cross-lingual: XLM-RoBERTa Large fine-tuned on "
         "PetroGeoNER corpus extracts terms + NER labels.\n\n"
         "Tests specialized NER model against general LLM."),
        ("P4 — TF-IDF", RED,
         "Statistical baseline: Term Frequency–Inverse Document Frequency "
         "on the 40-article corpus.\n\n"
         "Replicates traditional frequency-based methodology (Garcia et al., 2020)."),
    ]

    BW = Inches(5.85)
    BH = Inches(2.5)
    gap_x = Inches(0.35)
    gap_y = Inches(0.3)

    positions = [
        (MARGIN_X,         CONTENT_Y),
        (MARGIN_X + BW + gap_x, CONTENT_Y),
        (MARGIN_X,         CONTENT_Y + BH + gap_y),
        (MARGIN_X + BW + gap_x, CONTENT_Y + BH + gap_y),
    ]

    for (px, py), (label, color, detail) in zip(positions, pipelines):
        add_rect(slide, px, py, BW, BH, LIGHT_GRAY)
        add_rect(slide, px, py, BW, Inches(0.07), color)
        add_text(slide, px + Inches(0.15), py + Inches(0.12), BW - Inches(0.3), Inches(0.4),
                 label, size=16, bold=True, color=color)
        add_text(slide, px + Inches(0.15), py + Inches(0.6), BW - Inches(0.3), Inches(1.8),
                 detail, size=13.5, color=NEAR_BLACK, wrap=True)

    # Bottom note
    add_rect(slide, MARGIN_X, H - Inches(0.65), W - MARGIN_X * 2, Inches(0.38), PALE_BLUE)
    add_text(slide, MARGIN_X + Inches(0.1), H - Inches(0.6), W - MARGIN_X * 2 - Inches(0.2), Inches(0.35),
             "All 4 pipelines feed into the same unified NLD-centric classification module",
             size=13, bold=True, color=DARK_BLUE, align=PP_ALIGN.CENTER)


# ── SLIDE 6 — NLD Classification ─────────────────────────────────────────────
def slide_nld(prs):
    slide = _blank_slide(prs)
    add_header(slide, "Phase 3: NLD-Centric Classification Workflow",
               "Standardized across all pipelines — explicitly tests Lopes Junior (2025)")
    add_footer(slide)

    # Step boxes (shorter to leave room for examples)
    steps = [
        ("Step 1 — NLD Generation", MID_BLUE,
         'LLM role: "Senior geoscientist + ontology engineer"\n\n'
         'Aristotelian-style definition for each term:\n'
         '"X is a Y that Z"\n\n'
         "Concise, technically clear, canonical"),
        ("Step 2 — CoT Categorization", ORANGE,
         "Chain-of-Thought prompt: LLM analyzes\nterm + NLD together\n\n"
         "Assigns most specific category possible\n\n"
         "Fallback hierarchy:\nGeoReservoir  →  GeoCore  →  BFO"),
        ("Resulting Artifact", GREEN,
         "Formally instantiated into\nmulti-layered hierarchy:\n\n"
         "BFO → GeoCore → GeoReservoir\n\n"
         "Expert-validated taxonomy"),
    ]

    SW = Inches(3.6)
    SH = Inches(2.55)
    gap = Inches(0.45)
    total = 3 * SW + 2 * gap
    sx = (W - total) / 2

    for i, (label, color, detail) in enumerate(steps):
        px = sx + i * (SW + gap)
        py = CONTENT_Y + Inches(0.05)
        add_rect(slide, px, py, SW, SH, LIGHT_GRAY)
        add_rect(slide, px, py, SW, Inches(0.07), color)
        add_text(slide, px + Inches(0.15), py + Inches(0.13), SW - Inches(0.3), Inches(0.45),
                 label, size=15, bold=True, color=color)
        add_text(slide, px + Inches(0.15), py + Inches(0.65), SW - Inches(0.3), Inches(1.8),
                 detail, size=13, color=NEAR_BLACK, wrap=True)
        if i < 2:
            ax = px + SW + Inches(0.07)
            add_text(slide, ax, py + SH / 2 - Inches(0.2), gap - Inches(0.14), Inches(0.35),
                     "→", size=24, bold=True, color=MID_BLUE, align=PP_ALIGN.CENTER)

    # ── Concrete examples ──
    EY = CONTENT_Y + SH + Inches(0.35)
    add_text(slide, MARGIN_X, EY, W - MARGIN_X * 2, Inches(0.35),
             "Examples from the Pre-Salt corpus:", size=15, bold=True, color=DARK_BLUE)

    examples = [
        (
            "Coquina",
            "GeoReservoir:SedimentaryRock",
            MID_BLUE,
            '"Coquina is a sedimentary rock that is composed of cemented bioclastic '
            'fragments — mainly mollusk shells — deposited in high-energy shallow marine '
            'environments, serving as a primary Pre-Salt reservoir rock in Brazil."',
        ),
        (
            "Diagenesis",
            "GeoCore:GeologicalProcess",
            ORANGE,
            '"Diagenesis is a geological process that encompasses the physical, chemical, '
            'and biological changes affecting sedimentary material after deposition, '
            'including cementation, compaction, and dissolution, altering reservoir quality."',
        ),
        (
            "Porosity",
            "BFO:Quality",
            GREEN,
            '"Porosity is a quality of a rock that is measured by the ratio of void space '
            'to total rock volume, controlling fluid storage capacity and governing '
            'the economic viability of a petroleum reservoir."',
        ),
    ]

    EW = (W - 2 * MARGIN_X - 2 * Inches(0.25)) / 3
    EH = Inches(2.3)
    for k, (term, category, color, nld) in enumerate(examples):
        ex = MARGIN_X + k * (EW + Inches(0.25))
        ey = EY + Inches(0.42)
        add_rect(slide, ex, ey, EW, EH, PALE_BLUE)
        add_rect(slide, ex, ey, EW, Inches(0.06), color)
        # Term label
        add_text(slide, ex + Inches(0.12), ey + Inches(0.1), EW - Inches(0.2), Inches(0.32),
                 f"Term:  {term}", size=13, bold=True, color=color)
        # NLD text
        add_text(slide, ex + Inches(0.12), ey + Inches(0.46), EW - Inches(0.2), Inches(1.35),
                 nld, size=11.5, italic=True, color=NEAR_BLACK, wrap=True)
        # Category badge
        add_rect(slide, ex + Inches(0.1), ey + EH - Inches(0.45), EW - Inches(0.2), Inches(0.34), color)
        add_text(slide, ex + Inches(0.15), ey + EH - Inches(0.43), EW - Inches(0.3), Inches(0.3),
                 f"→  {category}", size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)


# ── SLIDE 7 — Expert Validation Design ───────────────────────────────────────
def slide_validation(prs):
    slide = _blank_slide(prs)
    add_header(slide, "Phase 4: Blind Multi-Expert Validation Design")
    add_footer(slide)

    # Experts box
    add_rect(slide, MARGIN_X, CONTENT_Y, Inches(5.0), Inches(1.8), PALE_BLUE)
    add_rect(slide, MARGIN_X, CONTENT_Y, Inches(0.08), Inches(1.8), MID_BLUE)
    add_text(slide, MARGIN_X + Inches(0.2), CONTENT_Y + Inches(0.1),
             Inches(4.7), Inches(0.4), "5 Domain Specialists", size=16, bold=True, color=DARK_BLUE)
    add_text(slide, MARGIN_X + Inches(0.2), CONTENT_Y + Inches(0.52),
             Inches(4.7), Inches(1.2),
             "PhD & MSc holders — up to 20 years experience\nIndustry + Academia (Brazilian Pre-Salt)\n"
             "Provided 100 reference terms before extraction (20 each)",
             size=13.5, color=NEAR_BLACK, wrap=True)

    # Sampling box
    add_rect(slide, MARGIN_X + Inches(5.3), CONTENT_Y, Inches(7.2), Inches(1.8), PALE_BLUE)
    add_rect(slide, MARGIN_X + Inches(5.3), CONTENT_Y, Inches(0.08), Inches(1.8), ORANGE)
    add_text(slide, MARGIN_X + Inches(5.5), CONTENT_Y + Inches(0.1),
             Inches(6.9), Inches(0.4), "Sampling Strategy", size=16, bold=True, color=DARK_BLUE)
    add_text(slide, MARGIN_X + Inches(5.5), CONTENT_Y + Inches(0.52),
             Inches(6.9), Inches(1.2),
             "100 top-ranked terms/pipeline × 4 pipelines = 400 entries\n"
             "Combined, anonymized, and randomized into a single list\n"
             "Blind: evaluators did not know which pipeline produced each term",
             size=13.5, color=NEAR_BLACK, wrap=True)

    # Three metric boxes
    metrics = [
        ("Term Relevance", MID_BLUE, "5 experts",
         "Relevant / Irrelevant / Invalid / Unknown\nSame specialists who provided reference terms"),
        ("NLD Accuracy", ORANGE, "3 experts",
         "Correct / Partially Correct / Incorrect\nAmbiguous / Wrong Context / Unknown"),
        ("Category Accuracy", GREEN, "3 experts",
         "Correct & Specific / Correct but Unspecific\nIncorrect / Unknown\nCaptures both accuracy and ontological depth"),
    ]

    MW = Inches(3.85)
    MH = Inches(2.85)
    MY = CONTENT_Y + Inches(2.1)
    mgap = Inches(0.28)

    for i, (label, color, who, detail) in enumerate(metrics):
        mx = MARGIN_X + i * (MW + mgap)
        add_rect(slide, mx, MY, MW, MH, LIGHT_GRAY)
        add_rect(slide, mx, MY, MW, Inches(0.07), color)
        add_text(slide, mx + Inches(0.15), MY + Inches(0.13), MW - Inches(0.3), Inches(0.42),
                 label, size=15, bold=True, color=color)
        add_text(slide, mx + Inches(0.15), MY + Inches(0.55), MW - Inches(0.3), Inches(0.35),
                 who, size=12, italic=True, color=MED_GRAY)
        add_text(slide, mx + Inches(0.15), MY + Inches(0.9), MW - Inches(0.3), Inches(1.85),
                 detail, size=13, color=NEAR_BLACK, wrap=True)


# ── SLIDE 8 — Results: Coverage & Relevance ──────────────────────────────────
def slide_results_coverage(prs):
    slide = _blank_slide(prs)
    add_header(slide, "Results — Term Coverage & Relevance")
    add_footer(slide)

    # Table
    headers = ["Pipeline", "Recall (%)", "Relevant (%)", "Irrelevant (%)", "Strictly Relevant\n(Consensus)"]
    rows = [
        ("LLM + Corpus",   "60",  "97.8",  "2.0",  "91 / 100"),
        ("LLM Generated",  "40",  "96.2",  "3.8",  "88 / 100"),
        ("NER + Corpus",   "32",  "86.6",  "11.8", "66 / 100"),
        ("TF-IDF",         "25",  "46.0",  "46.4", "25 / 100"),
    ]
    col_widths = [Inches(2.4), Inches(1.55), Inches(1.55), Inches(1.7), Inches(2.3)]
    row_h = Inches(0.5)
    table_x = MARGIN_X
    table_y = CONTENT_Y + Inches(0.1)

    # Header row
    cx = table_x
    for j, (hdr, cw) in enumerate(zip(headers, col_widths)):
        add_rect(slide, cx, table_y, cw, row_h, DARK_BLUE,
                 line=RGBColor(0x8F, 0xA8, 0xD4))
        add_text(slide, cx + Inches(0.07), table_y + Inches(0.04), cw - Inches(0.1), row_h - Inches(0.05),
                 hdr, size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        cx += cw

    row_colors = [
        RGBColor(0xE8, 0xF1, 0xFB),  # LLM+Corpus
        LIGHT_GRAY,
        LIGHT_GRAY,
        RGBColor(0xFD, 0xED, 0xED),  # TF-IDF red-ish
    ]
    for i, (row, rc) in enumerate(zip(rows, row_colors)):
        cy = table_y + (i + 1) * row_h
        cx = table_x
        for j, (cell, cw) in enumerate(zip(row, col_widths)):
            add_rect(slide, cx, cy, cw, row_h, rc, line=RGBColor(0xCC, 0xCC, 0xCC))
            c_bold = (j == 0) or (i == 0 and j in (1, 2, 4))
            c_color = MID_BLUE if (i == 0 and j != 0) else NEAR_BLACK
            add_text(slide, cx + Inches(0.07), cy + Inches(0.1), cw - Inches(0.1), row_h - Inches(0.15),
                     cell, size=13, bold=c_bold, color=c_color, align=PP_ALIGN.CENTER)
            cx += cw

    # Insight callouts
    KX = table_x + sum(col_widths) + Inches(0.35)
    KW = W - KX - MARGIN_X

    insights = [
        (MID_BLUE, "LLM+Corpus: highest recall AND precision"),
        (ORANGE,   "Corpus grounding mitigates LLM \"domain blindness\""),
        (RED,      "TF-IDF: 46.4% irrelevant noise — semantically blind"),
        (MED_GRAY, "NER: decent recall, but 33 'Mixed Feelings'"),
    ]
    for k, (color, txt) in enumerate(insights):
        ky = CONTENT_Y + Inches(0.1) + k * Inches(0.95)
        add_rect(slide, KX, ky, KW, Inches(0.82), LIGHT_GRAY)
        add_rect(slide, KX, ky, Inches(0.07), Inches(0.82), color)
        add_text(slide, KX + Inches(0.17), ky + Inches(0.12), KW - Inches(0.25), Inches(0.6),
                 txt, size=13.5, color=NEAR_BLACK, wrap=True)

    # Footer note
    add_rect(slide, MARGIN_X, H - Inches(0.75), W - MARGIN_X * 2, Inches(0.38), PALE_BLUE)
    add_text(slide, MARGIN_X + Inches(0.15), H - Inches(0.7),
             W - MARGIN_X * 2 - Inches(0.2), Inches(0.35),
             "Recall = % of expert-defined reference terms recovered by each pipeline",
             size=12, italic=True, color=MED_GRAY)


# ── SLIDE 9 — Results: NLD & Categorization ──────────────────────────────────
def slide_results_nld_cat(prs):
    slide = _blank_slide(prs)
    add_header(slide, "Results — NLD Quality & Taxonomic Categorization")
    add_footer(slide)

    # ── NLD table (left) ──
    add_text(slide, MARGIN_X, CONTENT_Y, Inches(6.0), Inches(0.4),
             "Natural Language Definition Accuracy (%)", size=15, bold=True, color=DARK_BLUE)

    nld_rows = [
        ("Correct",            "53.7", "66.0", "57.0", "34.7"),
        ("Partially Correct",  "34.0", "24.3", "22.3", "22.0"),
        ("Incorrect",          " 9.7", " 6.7", "15.3", " 9.0"),
        ("Ambiguous/Unknown",  " 2.0", " 2.6", " 4.0", "30.4"),
    ]
    nld_hdrs = ["Status", "LLM+C", "LLM-G", "NER+C", "TF-IDF"]
    nld_cw = [Inches(2.05), Inches(0.9), Inches(0.9), Inches(0.9), Inches(0.9)]
    rh = Inches(0.42)
    ty = CONTENT_Y + Inches(0.45)
    cx = MARGIN_X
    for j, (h, cw) in enumerate(zip(nld_hdrs, nld_cw)):
        add_rect(slide, cx, ty, cw, rh, DARK_BLUE, line=RGBColor(0x8F, 0xA8, 0xD4))
        add_text(slide, cx + Inches(0.05), ty + Inches(0.05), cw - Inches(0.07), rh,
                 h, size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        cx += cw
    for i, row in enumerate(nld_rows):
        cy = ty + (i + 1) * rh
        cx = MARGIN_X
        rc = RGBColor(0xE8, 0xF1, 0xFB) if i == 0 else LIGHT_GRAY
        for j, (cell, cw) in enumerate(zip(row, nld_cw)):
            add_rect(slide, cx, cy, cw, rh, rc, line=RGBColor(0xCC, 0xCC, 0xCC))
            add_text(slide, cx + Inches(0.05), cy + Inches(0.07), cw - Inches(0.07), rh - Inches(0.1),
                     cell, size=12, bold=(i == 0), color=NEAR_BLACK, align=PP_ALIGN.CENTER)
            cx += cw

    # NLD note
    nx = MARGIN_X
    ny = ty + 5 * rh + Inches(0.15)
    add_text(slide, nx, ny, Inches(5.7), Inches(0.75),
             "LLM-Generated leads on NLD accuracy — canonical terms have robust internal definitions.\n"
             "LLM+Corpus suffers context blindness: complex terms lack source grounding.",
             size=12, italic=True, color=MED_GRAY, wrap=True)

    # ── Category table (right) ──
    RX = MARGIN_X + Inches(6.3)
    RW = W - RX - MARGIN_X
    add_text(slide, RX, CONTENT_Y, RW, Inches(0.4),
             "Taxonomic Categorization — Final Majority Metrics (%)", size=15, bold=True, color=DARK_BLUE)

    cat_rows = [
        ("Majority Correct ✓",          "71", "66", "68", "36"),
        ("Majority Partial ✓",           "11", " 9", "10", " 5"),
        ("Disputed / Rejected ✗",        "18", "25", "22", "59"),
        ("Strict Consensus\n(Correct & Specific)", "39", "43", "23", "17"),
    ]
    cat_hdrs = ["Metric", "LLM+C", "LLM-G", "NER+C", "TF-IDF"]
    cat_cw = [Inches(2.55), Inches(0.82), Inches(0.82), Inches(0.82), Inches(0.82)]
    cx = RX
    for j, (h, cw) in enumerate(zip(cat_hdrs, cat_cw)):
        add_rect(slide, cx, ty, cw, rh, DARK_BLUE, line=RGBColor(0x8F, 0xA8, 0xD4))
        add_text(slide, cx + Inches(0.05), ty + Inches(0.05), cw - Inches(0.07), rh,
                 h, size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        cx += cw
    row_colors_cat = [
        RGBColor(0xE2, 0xF3, 0xE2),  # green - correct
        LIGHT_GRAY,
        RGBColor(0xFD, 0xED, 0xED),  # red-ish - rejected
        LIGHT_GRAY,
    ]
    for i, (row, rc) in enumerate(zip(cat_rows, row_colors_cat)):
        cy = ty + (i + 1) * rh
        cx = RX
        for j, (cell, cw) in enumerate(zip(row, cat_cw)):
            add_rect(slide, cx, cy, cw, rh, rc, line=RGBColor(0xCC, 0xCC, 0xCC))
            add_text(slide, cx + Inches(0.05), cy + Inches(0.06), cw - Inches(0.07), rh - Inches(0.08),
                     cell, size=12, bold=(j == 0),
                     color=GREEN if (i == 0 and j != 0) else (RED if (i == 2 and j != 0) else NEAR_BLACK),
                     align=PP_ALIGN.CENTER)
            cx += cw

    # winner callout
    WY = ty + 5 * rh + Inches(0.15)
    add_rect(slide, RX, WY, RW, Inches(0.75), PALE_BLUE)
    add_rect(slide, RX, WY, Inches(0.08), Inches(0.75), MID_BLUE)
    add_text(slide, RX + Inches(0.2), WY + Inches(0.1), RW - Inches(0.3), Inches(0.6),
             "LLM+Corpus: highest overall accuracy (71%) + lowest rejection (18%).\n"
             "Corpus acts as a decisive 'validity filter' for extracted terms.",
             size=12.5, color=DARK_BLUE, wrap=True)


# ── SLIDE 10 — Resulting Conceptual Artifact ─────────────────────────────────
def slide_artifact(prs):
    slide = _blank_slide(prs)
    add_header(slide, "Resulting Conceptual Artifact",
               "Expert-validated preliminary domain taxonomy — Brazilian Pre-Salt Petroleum Reservoirs")
    add_footer(slide)

    # Left: what it is
    LW = Inches(5.0)
    add_text(slide, MARGIN_X, CONTENT_Y, LW, Inches(0.4),
             "What was produced", size=16, bold=True, color=DARK_BLUE)

    bullets = [
        (0, "Terms with unanimous expert consensus (Relevant + Correct & Specific)"),
        (0, "Formally placed in BFO → GeoCore → GeoReservoir hierarchy"),
        (0, "Not yet axiomatized (no properties/relations yet)"),
        (0, "Foundation for an industrial petroleum similarity-search system"),
    ]
    add_bullets(slide, MARGIN_X, CONTENT_Y + Inches(0.45), LW, Inches(1.6), bullets, size=14)

    # Example hierarchy paths
    add_text(slide, MARGIN_X, CONTENT_Y + Inches(2.15), LW, Inches(0.4),
             "Example taxonomy paths", size=16, bold=True, color=DARK_BLUE)

    paths = [
        ("Shale",
         "BFO:Entity → Continuant → Indep. Continuant\n→ Material Entity → GeoCore:AmountOfRock\n→ GeoReservoir:SedimentaryRock → Shale",
         MID_BLUE),
        ("Diagenesis",
         "BFO:Entity → Occurrent → Process\n→ GeoCore:GeologicalProcess → Diagenesis",
         ORANGE),
        ("Porosity",
         "BFO:Entity → Continuant → Spec. Dep. Continuant\n→ BFO:Quality → Porosity",
         GREEN),
    ]
    for k, (term, path, color) in enumerate(paths):
        py = CONTENT_Y + Inches(2.6) + k * Inches(1.2)
        add_rect(slide, MARGIN_X, py, LW, Inches(1.1), LIGHT_GRAY)
        add_rect(slide, MARGIN_X, py, Inches(0.07), Inches(1.1), color)
        add_text(slide, MARGIN_X + Inches(0.15), py + Inches(0.05), LW - Inches(0.3), Inches(0.35),
                 term, size=14, bold=True, color=color)
        add_text(slide, MARGIN_X + Inches(0.15), py + Inches(0.42), LW - Inches(0.3), Inches(0.65),
                 path, size=11, color=NEAR_BLACK, wrap=True)

    # Right: stats + availability
    RX = MARGIN_X + LW + Inches(0.35)
    RW = W - RX - MARGIN_X
    add_text(slide, RX, CONTENT_Y, RW, Inches(0.4),
             "Scope of the taxonomy", size=16, bold=True, color=DARK_BLUE)

    stats = [
        ("Sedimentary Rocks", "Evaporite, Microbialite, Shale, Source Rock, Coquina…"),
        ("Geological Processes", "Diagenesis, Dissolution, Compaction, Erosion…"),
        ("Geological Properties", "Porosity, Permeability, Reservoir Quality, Alkalinity…"),
        ("Earth Fluids", "Hydrocarbon, Organic Matter"),
        ("Geological Contacts", "Onlap, Unconformity"),
        ("Time Intervals", "Aptian, Barremian, Hauterivian, Early Cretaceous…"),
    ]
    for k, (cat, examples) in enumerate(stats):
        sy = CONTENT_Y + Inches(0.5) + k * Inches(0.65)
        add_rect(slide, RX, sy, RW, Inches(0.6), LIGHT_GRAY)
        add_rect(slide, RX, sy, Inches(0.06), Inches(0.6), MID_BLUE)
        add_text(slide, RX + Inches(0.15), sy + Inches(0.03), RW - Inches(0.25), Inches(0.28),
                 cat, size=12, bold=True, color=DARK_BLUE)
        add_text(slide, RX + Inches(0.15), sy + Inches(0.3), RW - Inches(0.25), Inches(0.28),
                 examples, size=11, italic=True, color=MED_GRAY)

    # GitHub link
    gy = CONTENT_Y + 6 * Inches(0.65) + Inches(0.7)
    add_rect(slide, RX, gy, RW, Inches(0.55), PALE_BLUE)
    add_text(slide, RX + Inches(0.1), gy + Inches(0.08), RW - Inches(0.2), Inches(0.4),
             "github.com/BDI-UFRGS/PreSaltOntology", size=13, bold=True, color=MID_BLUE,
             align=PP_ALIGN.CENTER)


# ── SLIDE 11 — Limitations as Motivation ─────────────────────────────────────
def slide_limitations(prs):
    slide = _blank_slide(prs)
    add_header(slide, "What the Experiment Revealed — Open Problems",
               "These findings directly motivate our current, improved implementation")
    add_footer(slide)

    problems = [
        (
            "Context Blindness",
            RED,
            "NLDs generated without source context → generic / outdated definitions",
            "• LLM+Corpus: 34% 'Partially Correct' NLDs despite best term quality\n"
            "• Outdated paradigms propagated (e.g., homogeneous microbial origin of Pre-Salt carbonates)\n"
            "• General LLM's training data has temporal constraints\n"
            "→ Fix: provide relevant corpus passages at NLD generation time (RAG)"
        ),
        (
            "Ambiguity — Not-Strict Consensus Gap",
            ORANGE,
            "Expert disagreement on complex real-world terms reveals categorization limits",
            "• LLM+Corpus: 39 Strict vs 32 Majority-only correct (categorization)\n"
            "• LLM's reasoning potential constrained by lack of domain grounding\n"
            "• Real terms extracted from reports are inherently more ambiguous\n"
            "→ Fix: source context in NLD phase expected to elevate majority → strict consensus"
        ),
        (
            "Granularity Gap",
            MID_BLUE,
            "Automated extraction favors generic terms; compound terms and properties are under-extracted",
            "• Petrophysical properties (e.g., 'grain roundness') systematically missed\n"
            "• Prompt constraints prioritized classes, excluded numerical values\n"
            "• Experts expected more specific compound terms\n"
            "→ Fix: explicit prompt instruction to also extract properties and attributes"
        ),
    ]

    PW = (W - 2 * MARGIN_X - 2 * Inches(0.28)) / 3
    PH = Inches(4.1)

    for i, (title, color, summary, detail) in enumerate(problems):
        px = MARGIN_X + i * (PW + Inches(0.28))
        py = CONTENT_Y + Inches(0.1)
        add_rect(slide, px, py, PW, PH, LIGHT_GRAY)
        add_rect(slide, px, py, PW, Inches(0.07), color)
        add_text(slide, px + Inches(0.12), py + Inches(0.12), PW - Inches(0.24), Inches(0.42),
                 title, size=16, bold=True, color=color)
        add_rect(slide, px + Inches(0.1), py + Inches(0.6), PW - Inches(0.2), Inches(0.06), color)
        add_text(slide, px + Inches(0.12), py + Inches(0.72), PW - Inches(0.24), Inches(0.55),
                 summary, size=12.5, italic=True, color=NEAR_BLACK, wrap=True)
        add_text(slide, px + Inches(0.12), py + Inches(1.35), PW - Inches(0.24), Inches(2.65),
                 detail, size=12, color=NEAR_BLACK, wrap=True)

    # Bottom banner
    BY = CONTENT_Y + PH + Inches(0.3)
    add_rect(slide, MARGIN_X, BY, W - MARGIN_X * 2, Inches(0.65), PALE_BLUE)
    add_rect(slide, MARGIN_X, BY, Inches(0.08), Inches(0.65), ORANGE)
    add_text(slide, MARGIN_X + Inches(0.2), BY + Inches(0.1),
             W - MARGIN_X * 2 - Inches(0.35), Inches(0.5),
             "All three problems point to the same architectural fix: "
             "Retrieval-Augmented Generation (RAG) in the NLD generation phase.",
             size=14, bold=True, color=DARK_BLUE, wrap=True)


# ── SLIDE 12 — Current Work & Conclusion ─────────────────────────────────────
def slide_conclusion(prs):
    slide = _blank_slide(prs)
    add_header(slide, "Ongoing Work & Conclusion")
    add_footer(slide)

    # Left — Current work
    LW = Inches(6.0)
    add_text(slide, MARGIN_X, CONTENT_Y, LW, Inches(0.4),
             "Current Implementation (addressing all open problems)", size=16, bold=True, color=DARK_BLUE)

    current = [
        (0, "Hybrid RAG retrieval: dense (BGE-M3) + sparse (BM25) + cross-encoder reranker"),
        (0, "NLD generation grounded in source corpus passages"),
        (0, "Expanded corpus: 80 papers (2× the preliminary experiment)"),
        (0, "Ablation study: 4 conditions — Full RAG / No RAG / No NLD / Raw RAG"),
        (0, "Statistical expert evaluation: Wilcoxon, Friedman, ICC, Fleiss' kappa"),
        (0, "OWL export: complete taxonomy → Turtle RDF (Protégé-compatible)"),
    ]
    add_bullets(slide, MARGIN_X, CONTENT_Y + Inches(0.5), LW, Inches(2.8), current, size=14)

    # Right — Conclusions
    RX = MARGIN_X + LW + Inches(0.4)
    RW = W - RX - MARGIN_X
    add_text(slide, RX, CONTENT_Y, RW, Inches(0.4),
             "Conclusions from this paper", size=16, bold=True, color=DARK_BLUE)

    conclusions = [
        (GREEN, "LLM+Corpus is the strongest strategy: 60% recall, 97.8% precision, 71% correct categories"),
        (GREEN, "NLDs as a semantic pivot are effective: formal multi-layered classification without fine-tuning"),
        (GREEN, "No fine-tuning required: general-purpose LLMs can bootstrap domain ontology construction"),
        (ORANGE, "Context blindness is the principal remaining bottleneck"),
        (MID_BLUE, "RAG is the clear architectural next step"),
    ]
    for k, (color, txt) in enumerate(conclusions):
        cy = CONTENT_Y + Inches(0.5) + k * Inches(0.88)
        add_rect(slide, RX, cy, RW, Inches(0.74), LIGHT_GRAY)
        add_rect(slide, RX, cy, Inches(0.07), Inches(0.74), color)
        add_text(slide, RX + Inches(0.18), cy + Inches(0.1), RW - Inches(0.28), Inches(0.58),
                 txt, size=13, color=NEAR_BLACK, wrap=True)

    # Bottom thank you
    TY = H - Inches(1.05)
    add_rect(slide, 0, TY, W, Inches(1.05), DARK_BLUE)
    add_rect(slide, 0, TY, W, Inches(0.05), ORANGE)
    add_text(slide, MARGIN_X, TY + Inches(0.12), W - MARGIN_X * 2, Inches(0.45),
             "Thank you!   Questions welcome.", size=22, bold=True,
             color=WHITE, align=PP_ALIGN.CENTER)
    add_text(slide, MARGIN_X, TY + Inches(0.57), W - MARGIN_X * 2, Inches(0.38),
             "github.com/BDI-UFRGS/PreSaltOntology  ·  Funded by CAPES, CNPq, Petrobras",
             size=13, color=PALE_BLUE, align=PP_ALIGN.CENTER)


# ── Build & save ─────────────────────────────────────────────────────────────
def main():
    prs = Presentation()
    prs.slide_width  = W
    prs.slide_height = H

    slide_title(prs)
    slide_context(prs)
    slide_rq(prs)
    slide_method_overview(prs)
    slide_pipelines(prs)
    slide_nld(prs)
    slide_validation(prs)
    slide_results_coverage(prs)
    slide_results_nld_cat(prs)
    slide_artifact(prs)
    slide_limitations(prs)
    slide_conclusion(prs)

    out = r"c:\Users\marandrade\VSCode\Pipeline_2\docs\ICEIS2026_presentation_v2.pptx"
    prs.save(out)
    print(f"Saved: {out}  ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
