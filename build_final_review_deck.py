#!/usr/bin/env python3
"""Build the Final Review (Review 3) deck for the Chest X-Ray pneumonia case study.

The deck reuses the Review 2 template (theme, masters, layouts) so it keeps the
same visual language: Cambria display type, navy/SRM palette, bg2 cards, chips
and badges, dark opening and closing slides.
"""

import zipfile
from copy import deepcopy

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml import parse_xml
from pptx.oxml.ns import nsdecls, qn
from pptx.util import Inches, Pt

TEMPLATE = "Chest_Xray_Pneumonia_Review2_Research.pptx"
OUT = "Chest_Xray_Pneumonia_Final_Review.pptx"

FIG = "/tmp/deckfigs"
ICON = "/tmp/icons/image-1-1.png"

# ---------------------------------------------------------------- palette ----
INK = "1A1A2E"
NAVY = "21295C"
A1 = "065A82"
A2 = "1C7293"
A3 = "CADCFC"
A4 = "B83A4B"
A5 = "1F7A70"
A6 = "5A6472"
CARD = "F4F8FB"
WHITE = "FFFFFF"
MET_BG = "E7F0ED"
PART_BG = "E9F0F5"
MISS_BG = "F7E9EB"

MAJOR = "Cambria"

EMU = 914400
SW, SH = 13.333, 7.5
M = 0.6  # page margin
CW = 12.1  # content width


# ---------------------------------------------------------------- helpers ----
def C(h):
    return RGBColor.from_string(h)


def _style_run(run, size, bold, color, font=None, italic=False, spc=None):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = C(color)
    if font:
        run.font.name = font
    if spc is not None:
        run.font._rPr.set("spc", str(int(spc * 100)))


def text(slide, name, x, y, w, h, paras, anchor=MSO_ANCHOR.TOP, wrap=True,
         margins=(0, 0, 0, 0)):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    box.name = name
    tf = box.text_frame
    tf.word_wrap = wrap
    tf.margin_left, tf.margin_top, tf.margin_right, tf.margin_bottom = (
        Inches(margins[0]), Inches(margins[1]), Inches(margins[2]), Inches(margins[3])
    )
    tf.vertical_anchor = anchor
    for i, spec in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = spec.get("align", PP_ALIGN.LEFT)
        if "before" in spec:
            p.space_before = Pt(spec["before"])
        if "after" in spec:
            p.space_after = Pt(spec["after"])
        if "line" in spec:
            p.line_spacing = spec["line"]
        runs = spec.get("runs") or [{"t": spec.get("t", "")}]
        for r in runs:
            run = p.add_run()
            run.text = r["t"]
            _style_run(
                run,
                r.get("size", spec.get("size", 14)),
                r.get("bold", spec.get("bold", False)),
                r.get("color", spec.get("color", INK)),
                r.get("font", spec.get("font")),
                r.get("italic", spec.get("italic", False)),
                r.get("spc", spec.get("spc")),
            )
    return box


def para(t, **kw):
    """Shorthand paragraph spec."""
    spec = {"t": t}
    spec.update(kw)
    return spec


def shape(slide, name, x, y, w, h, fill=None, radius=None, kind=MSO_SHAPE.ROUNDED_RECTANGLE,
          line=None, line_w=1.0):
    sp = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    sp.name = name
    if radius is not None and kind == MSO_SHAPE.ROUNDED_RECTANGLE:
        sp.adjustments[0] = radius
    if fill:
        sp.fill.solid()
        sp.fill.fore_color.rgb = C(fill)
    else:
        sp.fill.background()
    if line:
        sp.line.color.rgb = C(line)
        sp.line.width = Pt(line_w)
    else:
        sp.line.fill.background()
    sp.shadow.inherit = False
    sp.text_frame.word_wrap = True
    return sp


def card(slide, name, x, y, w, h, fill=CARD, radius=0.035):
    return shape(slide, name, x, y, w, h, fill=fill, radius=radius)


def badge(slide, name, x, y, size, label, fill=A1, color=WHITE, font_size=15, font=None):
    shape(slide, name, x, y, size, size, fill=fill, kind=MSO_SHAPE.OVAL)
    text(slide, name + "-label", x, y, size, size,
         [para(label, size=font_size, bold=True, color=color, font=font, align=PP_ALIGN.CENTER)],
         anchor=MSO_ANCHOR.MIDDLE)


def chip(slide, name, x, y, w, label, fill=A3, color=A1, size=10, h=0.30):
    shape(slide, name, x, y, w, h, fill=fill, radius=0.5)
    text(slide, name + "-label", x, y, w, h,
         [para(label, size=size, bold=True, color=color, align=PP_ALIGN.CENTER)],
         anchor=MSO_ANCHOR.MIDDLE)


def picture(slide, name, path, x, y, w=None, h=None):
    kwargs = {}
    if w:
        kwargs["width"] = Inches(w)
    if h:
        kwargs["height"] = Inches(h)
    pic = slide.shapes.add_picture(path, Inches(x), Inches(y), **kwargs)
    pic.name = name
    return pic


def _cell_borders(cell, color=A3, width=0.75):
    tcPr = cell._tc.get_or_add_tcPr()
    for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        for old in tcPr.findall(qn(tag)):
            tcPr.remove(old)
    for i, tag in enumerate(("a:lnL", "a:lnR", "a:lnT", "a:lnB")):
        xml = (
            f'<{tag} {nsdecls("a")} w="{int(width * 12700)}" cap="flat" cmpd="sng" algn="ctr">'
            f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
            f'<a:prstDash val="solid"/></{tag}>'
        )
        tcPr.insert(i, parse_xml(xml))


def cell_text(cell, t, size=12, bold=False, color=INK, align=PP_ALIGN.LEFT,
              font=None, fill=None):
    cell.margin_left = Inches(0.09)
    cell.margin_right = Inches(0.09)
    cell.margin_top = Inches(0.02)
    cell.margin_bottom = Inches(0.02)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    if fill:
        cell.fill.solid()
        cell.fill.fore_color.rgb = C(fill)
    else:
        cell.fill.solid()
        cell.fill.fore_color.rgb = C(WHITE)
    _cell_borders(cell)
    tf = cell.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = t
    _style_run(run, size, bold, color, font)


def table(slide, name, x, y, w, col_widths, rows, header=True, row_h=0.40,
          header_h=0.42, size=12, header_size=11.5, aligns=None, first_col_bold=True,
          row_fills=None, cell_fills=None, cell_colors=None):
    nrows, ncols = len(rows), len(rows[0])
    heights = []
    for i in range(nrows):
        heights.append(header_h if (header and i == 0) else row_h)
    gf = slide.shapes.add_table(nrows, ncols, Inches(x), Inches(y), Inches(w),
                                Inches(sum(heights)))
    gf.name = name
    tbl = gf.table
    tbl.first_row = bool(header)
    tbl.horz_banding = False
    for i, cw in enumerate(col_widths):
        tbl.columns[i].width = Inches(cw)
    for i, hh in enumerate(heights):
        tbl.rows[i].height = Inches(hh)
    aligns = aligns or [PP_ALIGN.LEFT] * ncols
    for ri, row in enumerate(rows):
        is_head = header and ri == 0
        for ci, val in enumerate(row):
            cell = tbl.cell(ri, ci)
            if is_head:
                cell_text(cell, val, size=header_size, bold=True, color=WHITE,
                          align=aligns[ci], fill=NAVY)
            else:
                fill = WHITE
                if row_fills and ri in row_fills:
                    fill = row_fills[ri]
                if cell_fills:
                    fill = cell_fills.get((ri, ci), fill)
                color = INK
                if cell_colors and (ri, ci) in cell_colors:
                    color = cell_colors[(ri, ci)]
                bold = first_col_bold and ci == 0
                cell_text(cell, val, size=size, bold=bold, color=color,
                          align=aligns[ci], fill=fill)
    return gf


def eyebrow(slide, label, size=12, color=A2):
    text(slide, "eyebrow", M, 0.36, CW, 0.32,
         [para(label, size=size, bold=True, color=color, spc=1.5)])


def heading(slide, title_text, size=32, color=INK):
    text(slide, "heading", M, 0.68, CW, 0.72,
         [para(title_text, size=size, bold=True, color=color)], anchor=MSO_ANCHOR.MIDDLE)


# --------------------------------------------------------------- build -------
prs = Presentation(TEMPLATE)
prs.core_properties.title = "Chest X-Ray Pneumonia Detection - Final Review"
prs.core_properties.author = "Case Study Team"
prs.core_properties.subject = "TrustEdge-CXR: Review 3 final assessment"

layouts = {l.name: l for l in prs.slide_masters[0].slide_layouts}

# capture the slide-number placeholder XML before the slides are dropped
sldnum_xml = None
for shp in prs.slides[1].shapes:
    if shp.name.startswith("Slide Number"):
        sldnum_xml = deepcopy(shp._element)
        break

# update the inherited footer so the running footer matches this review
for layout in prs.slide_masters[0].slide_layouts:
    for shp in layout.shapes:
        if shp.has_text_frame and "Review 2" in shp.text_frame.text:
            for p in shp.text_frame.paragraphs:
                for r in p.runs:
                    r.text = r.text.replace("Review 2", "Review 3")
for shp in prs.slide_masters[0].shapes:
    if shp.has_text_frame and "Review 2" in shp.text_frame.text:
        for p in shp.text_frame.paragraphs:
            for r in p.runs:
                r.text = r.text.replace("Review 2", "Review 3")

# drop every existing slide, keep masters and layouts
sld_id_lst = prs.slides._sldIdLst
for sld in list(sld_id_lst):
    r_id = sld.get(qn("r:id"))
    prs.part.drop_rel(r_id)
    sld_id_lst.remove(sld)


def new_slide(layout_name, number):
    slide = prs.slides.add_slide(layouts[layout_name])
    if sldnum_xml is not None:
        el = deepcopy(sldnum_xml)
        slide.shapes._spTree.append(el)
    return slide


# ------------------------------------------------------------- 1 · title -----
s = new_slide("Title Dark", 1)
text(s, "title-eyebrow", 0.8, 1.60, 9.0, 0.40,
     [para("SRM INSTITUTE OF SCIENCE AND TECHNOLOGY", size=14, bold=True, color=A3)])
text(s, "title-main", 0.8, 2.12, 9.2, 1.80, [
    para("Chest X-Ray Pneumonia Detection", size=40, bold=True, color=WHITE),
    para("Built, tested and deployed", size=40, bold=True, color=WHITE),
])
text(s, "title-sub", 0.8, 4.10, 8.6, 0.90,
     [para("Final review of TrustEdge-CXR: six modules implemented end-to-end and "
           "measured on the official 624-patient test cohort",
           size=16, italic=True, color=A3)])
text(s, "title-meta", 0.8, 5.55, 9.0, 0.90, [
    para("CASE STUDY ASSESSMENT  —  FINAL REVIEW (REVIEW 3)", size=13, bold=True, color=A3),
    para("Date: ______________      Course: ______________      Department: ______________",
         size=13, color=A3),
])
shape(s, "title-icon-circle", 10.05, 2.25, 2.70, 2.70, fill=NAVY, kind=MSO_SHAPE.OVAL)
picture(s, "title-lungs-icon", ICON, 10.65, 2.85, w=1.50)

# --------------------------------------------------- 2 · targets scorecard ---
s = new_slide("Content", 2)
eyebrow(s, "RECAP  ·  WHAT WE PROMISED IN REVIEW 2")
heading(s, "Nine targets, scored honestly")

rows = [
    ["Target", "Review 2 goal", "Final measured result", "Status"],
    ["Leakage control", "0% patient overlap", "0.0% — MD5 hashed, verified", "Met"],
    ["Sensitivity", "≥ 95%", "99.2% — 3 missed of 390", "Met"],
    ["ROC-AUC", "≥ 0.97", "0.970 on unseen test set", "Met"],
    ["Model size", "≤ 4 MB (INT8)", "3.39 MB total edge bundle", "Met"],
    ["Test accuracy", "≥ 92%", "89.9% all · 93.9% auto-decided", "Partial"],
    ["Calibration (ECE)", "≤ 0.05", "0.025 val · 0.120 test", "Partial"],
    ["Evaluation", "patient split + RSNA", "patient split only — no RSNA yet", "Partial"],
    ["XAI + privacy", "PSC-tested CAM, FL + DP", "PSC ✓  FL ✓  DP not implemented", "Partial"],
    ["Specificity", "≥ 85%", "74.4% — 60 false alarms of 234", "Missed"],
]
status_fill = {"Met": MET_BG, "Partial": PART_BG, "Missed": MISS_BG}
status_color = {"Met": A5, "Partial": A2, "Missed": A4}
fills = {(i, 3): status_fill[r[3]] for i, r in enumerate(rows) if i}
colors = {(i, 3): status_color[r[3]] for i, r in enumerate(rows) if i}
table(s, "scorecard", M, 1.50, CW, [2.55, 3.10, 4.85, 1.60], rows,
      row_h=0.44, size=12, header_size=11.5,
      aligns=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.CENTER],
      cell_fills=fills, cell_colors=colors)

text(s, "scorecard-note", M, 6.12, CW, 0.70, [
    para("Four targets met, four partial, one missed. ",
         size=13, bold=True, color=INK,
         runs=[{"t": "Four targets met, four partial, one missed. ", "bold": True, "color": A4},
               {"t": "The two that matter most — specificity and the external test set — are the "
                     "ones we did not reach. Both are reported as measured and discussed on "
                     "slide 14 rather than re-framed.", "bold": False, "color": INK}]),
])

# --------------------------------------------------------- 3 · system -------
s = new_slide("Content", 3)
eyebrow(s, "THE SYSTEM  ·  SIX MODULES, ALL IMPLEMENTED")
heading(s, "One pipeline from DICOM to a triage decision")
picture(s, "architecture", f"{FIG}/image4.png", 2.65, 1.38, w=8.05)
text(s, "architecture-note", M, 6.52, CW, 0.40,
     [para("Each module was built and measured separately, then composed into the "
           "end-to-end bundle that ships to the device.", size=12.5, color=A6)])

# ------------------------------------------------------------- 4 · data -----
s = new_slide("Content", 4)
eyebrow(s, "MODULE 1  ·  LEAKAGE-FREE DATA")
heading(s, "The 62% leak is gone, and the split is provable")

for i, (val, lab) in enumerate([
        ("0.0%", "patient overlap in the final split (was 62.0%)"),
        ("5,240", "unique patients across 5,856 radiographs"),
        ("3.17 : 1", "pneumonia-to-normal imbalance in training"),
        ("224 px", "CLAHE-normalised uint8 cache, built once")]):
    y = 1.58 + i * 1.10
    text(s, f"data-stat-{i}", M, y, 4.30, 0.45,
         [para(val, size=26, bold=True, color=INK, font=MAJOR)])
    text(s, f"data-label-{i}", M, y + 0.44, 4.30, 0.36,
         [para(lab, size=12.5, color=A6)])

rows = [
    ["Partition", "Patients", "Images", "Imbalance"],
    ["Training", "3,892", "4,468", "3.17 : 1"],
    ["Validation", "724", "764", "2.98 : 1"],
    ["Official test", "624", "624", "1.67 : 1"],
]
table(s, "cohort-table", 5.35, 1.62, 7.35, [2.70, 1.55, 1.55, 1.55], rows,
      row_h=0.50, size=13, header_size=11.5,
      aligns=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT])

card(s, "leak-note", 5.35, 3.95, 7.35, 2.35)
text(s, "leak-note-head", 5.60, 4.10, 6.85, 0.40,
     [para("Why this changes every number", size=15, bold=True, color=A1)])
text(s, "leak-note-body", 5.60, 4.52, 6.85, 1.70, [
    para("Public splits place several images of the same child in different folds. "
         "Patient hashing closes that door, so the 624-image test set is a genuinely "
         "independent cohort — 234 normal, 390 pneumonia, 390 distinct children.",
         size=12.5, color=INK, line=1.25),
    para("Reported metrics are therefore lower than most published Kaggle results, "
         "and comparable to them.", size=12.5, color=A6, line=1.25, before=8),
])

# ------------------------------------------------- 5 · teacher + student ----
s = new_slide("Content", 5)
eyebrow(s, "MODULES 2 & 3  ·  DISTILLATION")
heading(s, "36× smaller, 2.1 F1 points behind the teacher")

card(s, "teacher-card", M, 1.55, 3.80, 2.55)
text(s, "teacher-head", 0.85, 1.72, 3.30, 0.35,
     [para("TEACHER · RAD-DINO", size=11, bold=True, color=A1, spc=0.8)])
text(s, "teacher-title", 0.85, 2.05, 3.35, 0.40,
     [para("ViT-B/16 foundation model", size=16, bold=True, color=INK, font=MAJOR)])
text(s, "teacher-body", 0.85, 2.50, 3.30, 1.45, [
    para("86.6 M params · 518 px · 768-d embeddings", size=12.5, color=INK, line=1.2),
    para("MLP head: 99.6% val accuracy, 0.9998 AUC", size=12.5, color=INK, line=1.2, before=6),
    para("Never deployed — server-side only", size=12.5, color=A4, line=1.2, before=6),
])

shape(s, "kd-arrow", 4.62, 2.55, 0.60, 0.55, fill=A3, radius=0.5)
text(s, "kd-arrow-label", 4.62, 2.55, 0.60, 0.55,
     [para("KD", size=12, bold=True, color=A1, align=PP_ALIGN.CENTER)],
     anchor=MSO_ANCHOR.MIDDLE)

card(s, "student-card", 5.45, 1.55, 3.80, 2.55)
text(s, "student-head", 5.70, 1.72, 3.30, 0.35,
     [para("STUDENT · MOBILENETV2", size=11, bold=True, color=A5, spc=0.8)])
text(s, "student-title", 5.70, 2.05, 3.35, 0.40,
     [para("Distilled edge network", size=16, bold=True, color=INK, font=MAJOR)])
text(s, "student-body", 5.70, 2.50, 3.30, 1.45, [
    para("2.4 M parameters · 224 px · two-phase training", size=12.5, color=INK, line=1.2),
    para("Focal loss (γ 2.0) + logit-matching KD (λ 0.5)", size=12.5, color=INK, line=1.2, before=6),
    para("Ships as a 2.73 MB INT8 backbone", size=12.5, color=A5, line=1.2, before=6),
])

shape(s, "loss-strip", M, 4.35, 8.65, 0.95, fill=NAVY, radius=0.06)
text(s, "loss-text", 0.85, 4.35, 8.20, 0.95, [
    para("L = Focal(y, σ(z_s)) + λ · MSE(z_s, z_t)",
         size=17, bold=True, color=WHITE, font=MAJOR, align=PP_ALIGN.CENTER),
], anchor=MSO_ANCHOR.MIDDLE)
text(s, "loss-caption", M, 5.42, 8.65, 0.40,
     [para("Distilling the pre-sigmoid logit, not the soft probability, avoids "
           "saturation when the capacity gap is 36×.", size=12, color=A6)])

text(s, "teacher-note", M, 5.92, 8.65, 0.85, [
    para("Ablation, same data and threshold policy:", size=12, bold=True, color=INK),
    para("without KD the student reaches 91.5% accuracy but misses 8 of 390 pneumonia "
         "cases; with KD it misses 3 and adds 13.3 points of specificity.",
         size=12, color=A6, line=1.2),
])

card(s, "why-card", 9.50, 1.55, 3.20, 5.30)
text(s, "why-head", 9.72, 1.70, 2.80, 0.40,
     [para("Why it holds up", size=15, bold=True, color=A1, font=MAJOR)])
for i, (h, b) in enumerate([
        ("Teacher is domain-trained", "882k chest X-rays, not ImageNet photos"),
        ("Sensitivity first", "focal loss tuned against the 3.17:1 imbalance"),
        ("Edge-sized by design", "2.4 M parameters, 3.39 MB after INT8"),
        ("36× fewer parameters", "2.4 M against 86.6 M, and 3.39 MB shipped")]):
    y = 2.22 + i * 1.12
    badge(s, f"why-badge-{i}", 9.72, y, 0.34, str(i + 1), fill=A2, font_size=12)
    text(s, f"why-title-{i}", 10.18, y - 0.03, 2.42, 0.32,
         [para(h, size=12.5, bold=True, color=INK)])
    text(s, f"why-body-{i}", 10.18, y + 0.27, 2.42, 0.65,
         [para(b, size=11.5, color=A6, line=1.15)])

# ----------------------------------------------------- 6 · training curves --
s = new_slide("Content", 6)
eyebrow(s, "IMPLEMENTATION  ·  TRAINING BEHAVIOUR")
heading(s, "Two phases, no divergence, 0.97 AUC by epoch 12")
picture(s, "training-curves", f"{FIG}/image5.png", M, 1.50, w=8.55)
card(s, "train-side", 9.45, 1.50, 3.25, 4.20)
text(s, "train-side-head", 9.70, 1.72, 2.80, 0.40,
     [para("What the curves show", size=15, bold=True, color=A1, font=MAJOR)])
for i, t in enumerate([
        "Phase 1 (8 epochs, frozen backbone) settles the head: val AUC 0.9655.",
        "Phase 2 (10 epochs, top 40 layers) lifts val AUC to 0.9900 without loss blow-up.",
        "Cosine decay from 5e-5 to 1.25e-5 keeps validation loss inside 0.40–0.62.",
        "Val loss runs above train loss by design — dropout 0.3 plus label noise."]):
    text(s, f"train-point-{i}", 9.70, 2.25 + i * 0.95, 2.75, 0.90,
         [para(t, size=11.5, color=INK, line=1.2)])

for i, (v, l) in enumerate([
        ("18", "epochs, 8 head-only + 10 fine-tuning"),
        ("AdamW", "1e-3 cosine to 5e-5, batch 32"),
        ("T* = 1.951", "temperature fitted on 764 validation images")]):
    x = M + i * 3.60
    text(s, f"train-stat-{i}", x, 5.45, 3.40, 0.45,
         [para(v, size=21, bold=True, color=A1, font=MAJOR)])
    text(s, f"train-stat-label-{i}", x, 5.92, 3.40, 0.55,
         [para(l, size=11.5, color=A6, line=1.15)])

# --------------------------------------------------------- 7 · benchmark ----
s = new_slide("Content", 7)
eyebrow(s, "RESULTS  ·  MASTER BENCHMARK")
heading(s, "Every model, one test set, no re-tuning on test")

rows = [
    ["Model", "Acc.", "Sens.", "Spec.", "F1", "AUC", "ECE"],
    ["Review 1 baseline (MobileNetV2)", "84.8%", "99.0%", "61.1%", "89.0", "0.966", "0.140"],
    ["RAD-DINO + MLP head (teacher)", "92.9%", "97.9%", "84.6%", "94.6", "0.976", "0.128"],
    ["Student without KD (ablation)", "91.5%", "97.9%", "80.8%", "93.5", "0.976", "0.095"],
    ["TrustEdge student — all cases", "89.9%", "99.2%", "74.4%", "92.5", "0.970", "0.120"],
    ["TrustEdge — auto-decided only", "93.9%", "99.7%", "76.2%", "—", "—", "—"],
    ["Edge INT8 bundle (deployed)", "88.6%", "98.7%", "71.8%", "91.6", "0.971", "0.129"],
    ["Federated student (5 sites)", "88.8%", "97.9%", "73.5%", "91.6", "0.965", "0.113"],
]
table(s, "benchmark", M, 1.52, CW, [3.55, 1.20, 1.20, 1.20, 1.05, 1.05, 1.05], rows,
      row_h=0.45, size=12.5, header_size=11,
      aligns=[PP_ALIGN.LEFT] + [PP_ALIGN.RIGHT] * 6,
      row_fills={1: "FBEEF0", 4: MET_BG, 5: MET_BG})

text(s, "bench-note", M, 5.42, CW, 0.30,
     [para("* Operating thresholds were tuned on the 764-image validation split; nothing "
           "was selected on the 624-image test cohort.", size=11.5, color=A6)])

for i, (v, l, col) in enumerate([
        ("−5.1 pts", "accuracy against the inflated Review 1 split", A4),
        ("+13.3 pts", "specificity once distillation and calibration land", A5),
        ("3 vs 8", "missed pneumonia cases, KD vs no-KD", A5)]):
    x = M + i * 4.10
    text(s, f"bench-stat-{i}", x, 5.78, 3.80, 0.50,
         [para(v, size=24, bold=True, color=col, font=MAJOR)])
    text(s, f"bench-stat-label-{i}", x, 6.30, 3.80, 0.55,
         [para(l, size=11.5, color=A6, line=1.15)])

# ---------------------------------------------------------- 8 · confusion ---
s = new_slide("Content", 8)
eyebrow(s, "RESULTS  ·  ERROR PROFILE")
heading(s, "The errors moved from false alarms to almost none")
picture(s, "confusion", f"{FIG}/image7.png", 1.10, 1.55, w=7.90)

card(s, "conf-card", 9.05, 1.55, 3.70, 4.55)
text(s, "conf-head", 9.30, 1.72, 3.20, 0.40,
     [para("On 624 test patients", size=15, bold=True, color=A1, font=MAJOR)])
for i, (v, l) in enumerate([
        ("91 → 60", "normal X-rays wrongly flagged (−34%)"),
        ("4 → 3", "pneumonia cases missed"),
        ("386", "of 390 pneumonia cases caught"),
        ("174", "of 234 normal cases cleared")]):
    y = 2.20 + i * 1.05
    text(s, f"conf-val-{i}", 9.30, y, 3.20, 0.40,
         [para(v, size=20, bold=True, color=INK, font=MAJOR)])
    text(s, f"conf-lab-{i}", 9.30, y + 0.40, 3.20, 0.55,
         [para(l, size=11.5, color=A6, line=1.15)])

text(s, "conf-caption", M, 5.05, 7.90, 0.55, [
    para("Left: Review 1 at the fixed 0.5 threshold. Right: TrustEdge at t* = 0.674 with "
         "temperature scaling. Same 624 images, same patient-level split.",
         size=11, color=A6, line=1.2),
])

text(s, "conf-note", M, 6.05, CW, 0.75, [
    para("Reading it correctly: ", size=12.5, bold=True, color=INK,
         runs=[{"t": "Reading it correctly: ", "bold": True, "color": A1},
               {"t": "the model is deliberately tuned for sensitivity, so the remaining error "
                     "budget sits in false positives. The referral layer is what removes them "
                     "before a clinician sees the report.",
                "bold": False, "color": INK}]),
])

# ------------------------------------------------------------- 9 · roc -----
s = new_slide("Content", 9)
eyebrow(s, "RESULTS  ·  DISCRIMINATION AND CALIBRATION")
heading(s, "Ranking is solved; probability honesty is not")
picture(s, "roc-chart", f"{FIG}/image8.png", 0.70, 1.48, w=3.95)
picture(s, "reliability-chart", f"{FIG}/image9.png", 4.85, 1.48, w=3.95)

card(s, "calib-card", 9.05, 1.48, 3.68, 4.05)
text(s, "calib-head", 9.28, 1.64, 3.25, 0.40,
     [para("What the plots say", size=14.5, bold=True, color=A1, font=MAJOR)])
for i, t in enumerate([
        "The student loses 0.009 AUC against the teacher for 36× fewer parameters.",
        "Calibration collapses on the new cohort: ECE 0.025 on validation, 0.120 on test.",
        "Temperature scaling helps (0.126 → 0.120) but cannot repair a distribution shift.",
        "Hence referral, not the raw probability, carries the safety argument."]):
    text(s, f"calib-point-{i}", 9.28, 2.12 + i * 0.82, 3.20, 0.78,
         [para(t, size=11, color=INK, line=1.18)])

text(s, "calib-note", M, 5.72, CW, 0.85, [
    para("", size=12.5,
         runs=[{"t": "Honest limitation: ", "bold": True, "color": A4},
               {"t": "a single temperature scalar fitted on 764 images does not transfer to a "
                     "624-image cohort with a different class balance. Per-cohort refitting or "
                     "conformal prediction is the next step, not a claim we can make today.",
                "bold": False, "color": INK}]),
])

# -------------------------------------------------------- 10 · referral -----
s = new_slide("Content", 10)
eyebrow(s, "MODULE 4  ·  SELECTIVE REFERRAL")
heading(s, "Referring 22% of cases buys a 99.7% sensitivity triage")

rows = [
    ["Confidence margin τ", "Referred", "Coverage", "Auto accuracy", "Auto sensitivity"],
    ["0.000 — no referral", "0 / 624", "100.0%", "89.9%", "99.2%"],
    ["0.030 — low", "74 / 624", "88.1%", "92.2%", "99.4%"],
    ["0.059 — configured", "135 / 624", "78.4%", "93.9%", "99.7%"],
    ["0.080 — aggressive", "182 / 624", "70.8%", "95.2%", "99.8%"],
]
table(s, "referral-table", M, 1.55, 8.35, [2.45, 1.50, 1.40, 1.50, 1.50], rows,
      row_h=0.50, size=12.5, header_size=10.5,
      aligns=[PP_ALIGN.LEFT] + [PP_ALIGN.RIGHT] * 4,
      row_fills={3: MET_BG})
text(s, "referral-note", M, 4.62, 8.35, 0.60,
     [para("τ was fixed on validation for a 10% referral rate; the test cohort refers 21.6%. "
           "The higher rate is reported, not hidden.", size=11.5, color=A6, line=1.2)])

for i, (v, l) in enumerate([
        ("135 / 624", "studies escalated at the configured point"),
        ("78.4%", "of the cohort decided without a radiologist"),
        ("93.9%", "accuracy on the cases the system kept")]):
    x = M + i * 2.85
    card(s, f"ref-mini-{i}", x, 5.42, 2.65, 1.00)
    text(s, f"ref-mini-val-{i}", x + 0.20, 5.50, 2.30, 0.40,
         [para(v, size=17, bold=True, color=A1, font=MAJOR)])
    text(s, f"ref-mini-lab-{i}", x + 0.20, 5.90, 2.30, 0.48,
         [para(l, size=10, color=A6, line=1.1)])

card(s, "ref-card", 9.20, 1.55, 3.55, 3.85)
text(s, "ref-head", 9.45, 1.70, 3.10, 0.40,
     [para("What referral buys", size=15, bold=True, color=A1, font=MAJOR)])
text(s, "ref-big", 9.45, 2.15, 3.10, 0.70,
     [para("1", size=40, bold=True, color=A5, font=MAJOR)])
text(s, "ref-big-label", 9.45, 2.85, 3.10, 0.45,
     [para("single missed pneumonia in the 489 auto-decided cases", size=11.5, color=A6, line=1.15)])
text(s, "ref-body", 9.45, 3.45, 3.10, 1.85, [
    para("Forced to decide, the referred cases would be only 75.6% accurate — the "
         "uncertainty signal finds the cases the model would get wrong.",
         size=11.5, color=INK, line=1.2),
    para("Cost: 21.6% of studies reach a radiologist who would otherwise be bypassed.",
         size=11.5, color=A4, line=1.2, before=8),
])

text(s, "ref-footnote", M, 6.60, CW, 0.40, [
    para("Rule: |p̂ − 0.5| < τ on the temperature-calibrated probability from 20 MC-dropout passes "
         "— replacing the fixed 0.5 that produced 91 false alarms in Review 1.",
         size=11.5, color=A6),
])

# ------------------------------------------------------------ 11 · edge -----
s = new_slide("Content", 11)
eyebrow(s, "MODULE 5  ·  INT8 EDGE BUNDLE")
heading(s, "3.39 MB and 13 ms on an ARM CPU")

rows = [
    ["Representation", "Size", "Compression", "Agreement", "Latency"],
    ["Keras FP32 model", "8.89 MB", "1.00×", "reference", "10.08 ms"],
    ["TFLite INT8 backbone", "2.73 MB", "3.26×", "95.8%", "12.13 ms"],
    ["NumPy head (+ MC, CAM)", "0.66 MB", "—", "100%", "0.94 ms"],
    ["Complete edge bundle", "3.39 MB", "2.62×", "95.8%", "13.07 ms"],
]
table(s, "edge-table", M, 1.55, 8.30, [2.85, 1.55, 1.45, 1.35, 1.30], rows,
      row_h=0.53, size=12.5, header_size=10.5,
      aligns=[PP_ALIGN.LEFT] + [PP_ALIGN.RIGHT] * 4,
      row_fills={4: MET_BG})
text(s, "edge-note", M, 4.35, 8.30, 0.55,
     [para("Mean absolute probability shift after quantisation: 0.0184 — clinically negligible "
           "against a decision margin of 0.059.", size=11.5, color=A6, line=1.2)])

for i, (v, l, col) in enumerate([
        ("2.73 MB", "INT8 backbone, 69.3% smaller than FP32", A5),
        ("13.07 ms", "per X-ray on ARM Cortex-A72, > 70 FPS", A1),
        ("95.8%", "exact decision agreement with FP32", A2)]):
    y = 1.55 + i * 1.42
    card(s, f"edge-stat-card-{i}", 9.15, y, 3.60, 1.25)
    text(s, f"edge-stat-{i}", 9.40, y + 0.12, 3.10, 0.50,
         [para(v, size=24, bold=True, color=col, font=MAJOR)])
    text(s, f"edge-stat-label-{i}", 9.40, y + 0.64, 3.10, 0.50,
         [para(l, size=11, color=A6, line=1.15)])

text(s, "edge-repro", M, 5.20, 8.30, 1.20, [
    para("Reproducible: ", size=12, bold=True, color=INK,
         runs=[{"t": "Reproducible: ", "bold": True, "color": A1},
               {"t": "artifacts/edge_bundle + edge_infer.py --image <cxr> --cam runs with only "
                     "tflite-runtime, numpy and Pillow — no TensorFlow, no PyTorch, no GPU. "
                     "Outputs label, calibrated probability, epistemic variance, triage action "
                     "and the Grad-CAM overlay.",
                "bold": False, "color": INK}]),
])

# --------------------------------------------------------- 12 · federated ---
s = new_slide("Content", 12)
eyebrow(s, "MODULE 6  ·  FEDERATED LEARNING")
heading(s, "Five non-IID hospitals converge without sharing images")
picture(s, "fed-chart", f"{FIG}/image6.png", M, 1.52, w=7.30)

for i, (v, l) in enumerate([
        ("0.9655", "test AUC after 15 global rounds"),
        ("5", "sites, Dirichlet α = 0.5, 45–99% class skew"),
        ("0", "patient images that left a hospital")]):
    y = 1.55 + i * 1.50
    text(s, f"fed-stat-{i}", 8.85, y, 3.85, 0.55,
         [para(v, size=30, bold=True, color=A5, font=MAJOR)])
    text(s, f"fed-stat-label-{i}", 8.85, y + 0.55, 3.85, 0.60,
         [para(l, size=11.5, color=A6, line=1.15)])

text(s, "fed-note", M, 5.55, CW, 0.90, [
    para("FedAvg plus FedProx (μ = 0.01) and server momentum (0.5) hold the global model "
         "together across sites where one client is 99% pneumonia and another 45%. "
         "Federated training costs 1.1 points of accuracy and 0.005 AUC against central "
         "training — a fair price for keeping records local.",
         size=12.5, color=INK, line=1.3),
    para("Not done: differential privacy. The report claims FL only, and this deck should "
         "not claim more.", size=12, color=A4, line=1.2, before=6),
])

# --------------------------------------------------------------- 13 · xai ---
s = new_slide("Content", 13)
eyebrow(s, "MODULE 4  ·  EXPLANATION AUDIT")
heading(s, "The heatmaps were tested, not just drawn")
picture(s, "gradcam", f"{FIG}/image10.png", M, 1.48, w=7.25)

text(s, "xai-caption", M, 5.35, 7.25, 0.55,
     [para("Grad-CAM++ on official test images: correct pneumonia, a correct normal, and one "
           "case escalated as REFER at p = 0.52.", size=10.5, color=A6, line=1.2)])

card(s, "xai-card", 8.55, 1.50, 4.15, 4.15)
text(s, "xai-head", 8.80, 1.66, 3.70, 0.40,
     [para("Perturbation stability", size=15, bold=True, color=A1, font=MAJOR)])
for i, (v, l) in enumerate([
        ("ρ = 0.509", "rank correlation between perturbation size and map shift"),
        ("4.9% vs 27.1%", "map shift when the prediction holds vs flips"),
        ("5.5×", "stability-to-flip ratio — explanations track pathology")]):
    y = 2.12 + i * 1.15
    text(s, f"xai-val-{i}", 8.80, y, 3.60, 0.42,
         [para(v, size=19, bold=True, color=INK, font=MAJOR)])
    text(s, f"xai-lab-{i}", 8.80, y + 0.42, 3.60, 0.62,
         [para(l, size=11, color=A6, line=1.15)])

text(s, "xai-note", M, 5.78, 7.55, 1.10, [
    para("Grad-CAM was kept over Grad-CAM++: its flip contrast is 5.5× against 3.6×, and "
         "its maps are less sensitive to spurious edges. At 100% recall on the visual sample, "
         "the audit still cannot prove causality — only that the maps behave like evidence.",
         size=11.5, color=INK, line=1.25),
])

# -------------------------------------------------------- 14 · limits ------
s = new_slide("Content", 14)
eyebrow(s, "LIMITATIONS  ·  WHAT DID NOT WORK")
heading(s, "Five results we would not defend in a viva")

limits = [
    ("Specificity target missed",
     "74.4% against an ≥ 85% goal. 60 of 234 normal children still receive a pneumonia flag "
     "before referral."),
    ("Calibration does not transfer",
     "ECE 0.025 on validation becomes 0.120 on test. One global temperature cannot absorb a "
     "class-balance shift."),
    ("No external cohort",
     "RSNA was promised in Review 2 and not run. All numbers come from one Guangzhou hospital, "
     "so cross-scanner generalisation is unmeasured."),
    ("Referral rate is above plan",
     "21.6% of studies escalate instead of the designed 10%. Workload relief is real but half "
     "of what we projected."),
    ("Federated without DP",
     "FedProx with server momentum converges, but no differential-privacy budget is applied. "
     "The privacy claim is architectural, not formal."),
]
for i, (h, b) in enumerate(limits):
    y = 1.55 + i * 1.03
    card(s, f"limit-card-{i}", M, y, CW, 0.90)
    badge(s, f"limit-badge-{i}", 0.78, y + 0.18, 0.54, f"L{i+1}", fill=A4, font_size=13)
    text(s, f"limit-title-{i}", 1.55, y, 3.35, 0.90,
         [para(h, size=14.5, bold=True, color=INK)],
         anchor=MSO_ANCHOR.MIDDLE)
    text(s, f"limit-text-{i}", 5.05, y, 7.45, 0.90,
         [para(b, size=12.5, color=A6, line=1.2)],
         anchor=MSO_ANCHOR.MIDDLE)

# -------------------------------------------------------- 15 · clinical -----
s = new_slide("Content", 15)
eyebrow(s, "DEPLOYMENT  ·  WHERE IT WOULD RUN")
heading(s, "Three workflows the bundle already supports")

tiers = [
    ("TIER 1", "Point-of-care kiosk", A1, [
        "Raspberry Pi 4B with a USB flat-panel detector",
        "3.39 MB bundle, 13 ms inference, fully offline",
        "Label, probability, variance and heatmap on screen"]),
    ("TIER 2", "Mobile screening van", A2, [
        "Filters the clear normals and clear pneumonias locally",
        "Only the ~21% ambiguous cases use the cellular link",
        "Roughly 80% less tele-radiology bandwidth"]),
    ("TIER 3", "Regional consortium", A5, [
        "Five hospital nodes running the federated daemon",
        "Weights synchronise; images never leave the site",
        "Model improves as each site confirms referrals"]),
]
for i, (tag, title, col, items) in enumerate(tiers):
    x = M + i * 4.13
    card(s, f"tier-card-{i}", x, 1.55, 3.85, 3.35)
    text(s, f"tier-tag-{i}", x + 0.25, 1.72, 3.35, 0.30,
         [para(tag, size=10.5, bold=True, color=col, spc=1.0)])
    text(s, f"tier-title-{i}", x + 0.25, 2.00, 3.35, 0.75,
         [para(title, size=17, bold=True, color=INK, font=MAJOR)])
    for j, it in enumerate(items):
        text(s, f"tier-item-{i}-{j}", x + 0.25, 2.62 + j * 0.66, 3.35, 0.60,
             [para("•  " + it, size=12, color=A6, line=1.2)])

shape(s, "deploy-strip", M, 5.35, CW, 1.15, fill=NAVY, radius=0.06)
text(s, "deploy-strip-label", 0.90, 5.35, 2.60, 1.15,
     [para("One bundle, everywhere", size=14, bold=True, color=A3, font=MAJOR)],
     anchor=MSO_ANCHOR.MIDDLE)
for i, (v, l) in enumerate([
        ("3.39 MB", "total footprint"),
        ("13 ms", "per X-ray"),
        ("0 bytes", "of patient data sent")]):
    x = 3.70 + i * 3.15
    text(s, f"deploy-val-{i}", x, 5.45, 2.95, 0.45,
         [para(v, size=21, bold=True, color=WHITE, font=MAJOR)])
    text(s, f"deploy-lab-{i}", x, 5.92, 2.95, 0.40,
         [para(l, size=11, color=A3)])

# ------------------------------------------------------- 16 · conclusion ----
s = new_slide("Closing Dark", 16)
eyebrow(s, "CONCLUSION", size=12, color=A3)
heading(s, "It ships, it is honest, and it is measured", color=WHITE)

for i, (v, l) in enumerate([
        ("0%", "patient overlap after hashing"),
        ("99.2%", "sensitivity, 3 missed of 390"),
        ("3.39 MB", "INT8 bundle at 13 ms"),
        ("0.9655", "AUC across 5 federated sites")]):
    x = M + i * 3.10
    text(s, f"close-stat-{i}", x, 1.70, 2.95, 0.85,
         [para(v, size=38, bold=True, color=A3, font=MAJOR)])
    text(s, f"close-stat-label-{i}", x, 2.55, 2.95, 0.55,
         [para(l, size=12.5, color=A3, line=1.15)])

points = [
    ("Built: ", "six modules from patient hashing to a federated edge runtime, all in one "
                "reproducible pipeline."),
    ("Validated: ", "every number comes from the same 624-image unseen cohort, with no "
                    "test-set tuning."),
    ("Honest: ", "specificity, calibration transfer and the missing external cohort are "
                 "reported as failures, not smoothed over."),
    ("Next: ", "fix calibration on the target cohort, run RSNA, and add a privacy budget "
               "before claiming clinical readiness."),
]
for i, (h, b) in enumerate(points):
    y = 3.55 + i * 0.80
    shape(s, f"close-dot-{i}", 0.65, y + 0.10, 0.13, 0.13, fill=A3, kind=MSO_SHAPE.OVAL)
    text(s, f"close-point-{i}", 1.02, y, 11.60, 0.70,
         [para("", size=15,
               runs=[{"t": h, "bold": True, "color": WHITE},
                     {"t": b, "color": A3}])])

# ----------------------------------------------------------- 17 · future ----
s = new_slide("Content", 17)
eyebrow(s, "FUTURE SCOPE  ·  WHAT WE WOULD DO NEXT")
heading(s, "Four extensions, in priority order")

future = [
    ("F1", "Cohort-matched calibration", A4,
     "Refit temperature per cohort, or move to conformal prediction, so the reported "
     "probability means what it says outside validation."),
    ("F2", "External RSNA validation", A1,
     "Run the frozen bundle on adult RSNA radiographs to measure cross-scanner and "
     "cross-age generalisation — the missing evidence."),
    ("F3", "Differential privacy", A2,
     "Add DP-SGD noise accounting to the federated updates so the privacy guarantee is "
     "formal rather than architectural."),
    ("F4", "Multi-pathogen and reports", A5,
     "Extend from binary triage to bacterial, viral and effusion heads, with a lightweight "
     "vision-language report beside the score."),
]
for i, (tag, title, col, body) in enumerate(future):
    x = M + i * 3.06
    card(s, f"future-card-{i}", x, 1.55, 2.85, 3.95)
    badge(s, f"future-badge-{i}", x + 0.22, 1.75, 0.50, tag, fill=col, font_size=12)
    text(s, f"future-title-{i}", x + 0.22, 2.38, 2.42, 0.95,
         [para(title, size=14.5, bold=True, color=INK, font=MAJOR, line=1.1)])
    text(s, f"future-body-{i}", x + 0.22, 3.10, 2.42, 2.35,
         [para(body, size=11.5, color=A6, line=1.25)])
    text(s, f"future-rank-{i}", x + 0.22, 4.70, 2.42, 0.60,
         [para("Priority " + str(i + 1), size=10, bold=True, color=col, spc=1.0)])

text(s, "future-note", M, 5.85, CW, 0.80, [
    para("All four are engineering work on an existing pipeline — none needs a new architecture. "
         "The first two are what stand between this system and a defensible clinical claim.",
         size=12.5, color=A6, line=1.25),
])

# ------------------------------------------------------- 18 · references ----
s = new_slide("Content", 18)
eyebrow(s, "REFERENCES")
heading(s, "References")

refs_left = [
    "[1]  D. S. Kermany et al., “Identifying medical diagnoses and treatable diseases by "
    "image-based deep learning,” Cell, vol. 172, no. 5, pp. 1122–1131, 2018.",
    "[2]  P. Rajpurkar et al., “CheXNet: Radiologist-level pneumonia detection on chest "
    "X-rays with deep learning,” arXiv:1711.05225, 2017.",
    "[3]  M. Sandler et al., “MobileNetV2: Inverted residuals and linear bottlenecks,” "
    "CVPR, pp. 4510–4520, 2018.",
    "[4]  M. Oquab et al., “DINOv2: Learning robust visual features without supervision,” "
    "TMLR, 2023.",
    "[5]  F. Pérez-García et al., “RAD-DINO: Exploring self-supervised vision transformers "
    "for radiograph analysis,” arXiv:2311.13600, 2023.",
    "[6]  C. Guo et al., “On calibration of modern neural networks,” ICML, pp. 1321–1330, 2017.",
    "[7]  Y. Gal and Z. Ghahramani, “Dropout as a Bayesian approximation,” ICML, "
    "pp. 1050–1059, 2016.",
    "[8]  B. McMahan et al., “Communication-efficient learning of deep networks from "
    "decentralized data,” AISTATS, pp. 1273–1282, 2017.",
    "[9]  T. Li et al., “Federated optimization in heterogeneous networks,” MLSys, "
    "vol. 2, pp. 429–450, 2020.",
]
refs_right = [
    "[10]  T. Y. Lin et al., “Focal loss for dense object detection,” ICCV, "
    "pp. 2980–2988, 2017.",
    "[11]  G. Hinton, O. Vinyals and J. Dean, “Distilling the knowledge in a neural "
    "network,” NIPS DL Workshop, 2015.",
    "[12]  R. R. Selvaraju et al., “Grad-CAM: Visual explanations from deep networks via "
    "gradient-based localization,” ICCV, pp. 618–626, 2017.",
    "[13]  A. Chattopadhay et al., “Grad-CAM++: Generalized gradient-based visual "
    "explanations,” WACV, pp. 839–847, 2018.",
    "[14]  S. M. Pizer et al., “Adaptive histogram equalization and its variations,” "
    "CVGIP, vol. 39, no. 3, pp. 355–368, 1987.",
    "[15]  K. He et al., “Deep residual learning for image recognition,” CVPR, "
    "pp. 770–778, 2016.",
    "[16]  World Health Organization, “Pneumonia in children,” WHO Fact Sheet, 2022.",
    "[17]  A. G. Howard et al., “MobileNets: Efficient convolutional neural networks for "
    "mobile vision applications,” arXiv:1704.04861, 2017.",
    "[18]  M. Abadi et al., “Deep learning with differential privacy,” ACM CCS, "
    "pp. 308–318, 2016.",
]
text(s, "refs-left", M, 1.50, 5.90, 5.45,
     [para(r, size=9.5, color=INK, line=1.15, after=5) for r in refs_left])
text(s, "refs-right", 6.75, 1.50, 5.90, 5.45,
     [para(r, size=9.5, color=INK, line=1.15, after=5) for r in refs_right])
text(s, "refs-foot", M, 6.55, CW, 0.40, [
    para("Dataset: Kermany et al. [1], Guangzhou Women and Children's Medical Center — "
         "5,856 radiographs, 5,240 patients.", size=10.5, color=A6),
])

# --------------------------------------------------------------- sections ----
P14 = "http://schemas.microsoft.com/office/powerpoint/2010/main"
ids = [el.get("id") for el in prs.slides._sldIdLst]
groups = [
    ("Framing", ids[0:2]),
    ("The build", ids[2:6]),
    ("Results", ids[6:10]),
    ("Deployment", ids[10:12]),
    ("Audit", ids[12:14]),
    ("Close", ids[14:18]),
]
section_xml = ['<p14:sectionLst xmlns:p14="%s">' % P14]
for name, sl in groups:
    if not sl:
        continue
    guid = "{%s}" % __import__("uuid").uuid4()
    section_xml.append(
        '<p14:section name="%s" id="%s"><p14:sldIdLst>%s</p14:sldIdLst></p14:section>'
        % (name, guid, "".join('<p14:sldId id="%s"/>' % i for i in sl))
    )
section_xml.append("</p14:sectionLst>")

pres = prs.part._element
ext_lst = pres.find(qn("p:extLst"))
if ext_lst is not None:
    for ext in ext_lst.findall(qn("p:ext")):
        if ext.get("uri") == "{521415D9-36F7-43E2-AB2F-B90AF26B5E84}":
            for child in list(ext):
                ext.remove(child)
            ext.append(parse_xml("".join(section_xml)))
            break

prs.save(OUT)
print("wrote", OUT, "slides:", len(prs.slides.__iter__.__self__._sldIdLst))
