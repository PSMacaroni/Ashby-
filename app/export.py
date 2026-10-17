"""
Exports for one slide, made when someone clicks a download button:
  - map_png:    a labeled picture of the map (title, map, color key), e.g. for the slide deck
  - pdf_report: a one-page PDF report (map, tissue breakdown, flags, findings, risk regions,
                where the AI is unsure, caveats)
Both only restate numbers the app already shows.
"""
import io
from datetime import date

from fpdf import FPDF
from fpdf.enums import XPos, YPos
from PIL import Image, ImageDraw, ImageFont

import explain
import tile_map
from common import CAVEATS, CLASSES, COLORS, NAMES, RISK_LEVELS, SHORT_NAMES, pct

MODE_TITLES = {"tissue": "Tissue map", "confidence": "How sure the AI is", "risk": "Risk levels"}
FOOTER = "Decision support only, not a diagnosis."


def hex_rgb(color):
    return tile_map.hex_rgb(color)


# ---------------------------------------------------------------- map picture

def map_picture(slide, mode="tissue", opacity=0.5):
    """The map with a title above and a color key below, as a PIL image."""
    data = tile_map.prepare(slide)
    s = data["summary"]
    img = tile_map.map_image(slide, mode, opacity)
    if img.width < 1400:                        # make small thumbnails bigger, keeping tile edges crisp
        k = 2
        img = img.resize((img.width * k, img.height * k), Image.NEAREST)

    pad = 28
    width = img.width + 2 * pad
    title_font = ImageFont.load_default(size=32)
    font = ImageFont.load_default(size=22)
    small = ImageFont.load_default(size=19)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    # lay out the color key in rows that fit the width
    rows, row, x = [], [], pad
    for label, color in tile_map.legend_items(data, mode):
        w = 26 + 8 + probe.textlength(label, font=font) + 30
        if row and x + w > width - pad:
            rows.append(row)
            row, x = [], pad
        row.append((label, color, x))
        x += w
    if row:
        rows.append(row)

    header, key_h, footer = 92, 38 * len(rows) + 16, 44
    canvas = Image.new("RGB", (width, header + img.height + key_h + footer), "white")
    d = ImageDraw.Draw(canvas)
    d.text((pad, 18), f"{slide.name} - {MODE_TITLES.get(mode, mode)}", fill=(20, 20, 20), font=title_font)
    model = explain.get(s, "model.name", "model not recorded")
    d.text((pad, 58), f"Case {slide.case_id} | Slide {slide.slide_id} | {model}", fill=(90, 90, 90), font=small)
    canvas.paste(img, (pad, header))

    y = header + img.height + 14
    for row in rows:
        for label, color, x in row:
            d.rectangle([x, y + 2, x + 26, y + 28], fill=hex_rgb(color), outline=(60, 60, 60))
            d.text((x + 34, y + 2), label, fill=(30, 30, 30), font=font)
        y += 38
    note = FOOTER + (" DEMO slide: fake data." if slide.is_demo else "") + \
        f" Colon Tumor Front Analyzer, {date.today():%Y-%m-%d}."
    d.text((pad, canvas.height - footer + 10), note, fill=(110, 110, 110), font=small)
    return canvas


def map_png(slide, mode="tissue", opacity=0.5):
    buf = io.BytesIO()
    map_picture(slide, mode, opacity).save(buf, "PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- one-page PDF report

# The PDF's built-in fonts only know Latin-1 characters, so swap out the few others we use
SWAPS = {"→": "->", "—": "-", "–": "-", "≥": ">=", "≤": "<=", "α": "alpha", "⚠": "!", "‘": "'", "’": "'",
         "“": '"', "”": '"', "…": "..."}


def latin1(text):
    for a, b in SWAPS.items():
        text = text.replace(a, b)
    return text.encode("latin-1", "replace").decode("latin-1")


class Report(FPDF):
    BOTTOM = 280          # mm: sections stop here (moved up to keep room for the caveats)

    def heading(self, text):
        self.set_font("Helvetica", "B", 10.5)
        self.set_text_color(14, 84, 99)
        self.cell(0, 6, latin1(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(30, 30, 30)

    def bullets(self, lines, size=8.5, max_lines=None):
        """Write lines as bullets; stops (and says so) if the page is full."""
        self.set_font("Helvetica", "", size)
        for i, line in enumerate(lines):
            if self.get_y() > self.BOTTOM - 5 or (max_lines and i >= max_lines):
                self.set_font("Helvetica", "I", size)
                self.cell(0, 4.2, f"+ {len(lines) - i} more in the app", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                break
            self.set_x(self.l_margin + 1.5)
            self.multi_cell(self.epw - 1.5, 4.2, latin1(f"- {line}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1.8)

    def section(self, title, lines, **kw):
        if self.get_y() > self.BOTTOM - 14:     # not enough room for a heading and a line
            return
        self.heading(title)
        self.bullets(lines, **kw)


def pdf_report(slide, summary_text=None):
    """
    A one-page PDF for a slide. summary_text: optional plain-language paragraph (e.g. Lumen's
    rewording); without it, the key findings are listed as they are.
    """
    data = tile_map.prepare(slide)
    s, g = data["summary"], data["geom"]
    pdf = Report("P", "mm", "A4")
    pdf.set_auto_page_break(False)
    pdf.set_margins(12, 12, 12)
    pdf.add_page()

    # header band
    pdf.set_fill_color(14, 84, 99)
    pdf.rect(0, 0, 210, 16, style="F")
    pdf.set_xy(12, 4.5)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(120, 7, "Colon Tumor Front Analyzer  |  slide report")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 7, f"{date.today():%Y-%m-%d}", align="R")

    # title + details
    pdf.set_xy(12, 20)
    pdf.set_text_color(20, 20, 20)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 8, latin1(slide.name), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(90, 90, 90)
    model = explain.get(s, "model.name", "not recorded")
    pdf.multi_cell(0, 4.2, latin1(f"Case {slide.case_id}  |  Slide {slide.slide_id}  |  Entered {slide.entered}  |  "
                                  f"{int(data['ok'].sum()):,} analyzed squares of about {g['tile_um']:.0f} µm  |  "
                                  f"Model: {model}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    if slide.is_demo:
        pdf.ln(1)
        pdf.set_fill_color(255, 236, 179)
        pdf.set_text_color(90, 70, 0)
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.cell(0, 6, "  DEMO slide: fake data made to build the app. Every number here is made up.",
                 fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2.5)

    # map (left) + tissue breakdown and AI confidence (right)
    top = pdf.get_y()
    img = tile_map.map_image(slide, "tissue", 0.5)
    map_w = 118
    map_h = min(82, map_w * img.height / img.width)
    map_w = map_h * img.width / img.height
    jpg = io.BytesIO()
    img.save(jpg, "JPEG", quality=90)          # JPEG keeps the PDF small
    pdf.image(jpg, x=12, y=top, w=map_w, h=map_h)
    pdf.set_draw_color(200, 200, 200)
    pdf.rect(12, top, map_w, map_h)

    x = 12 + map_w + 6
    col_w = 210 - 12 - x
    pdf.set_xy(x, top)
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.set_text_color(14, 84, 99)
    pdf.cell(col_w, 5, "Tissue (share of analyzed squares)", new_x=XPos.LEFT, new_y=YPos.NEXT)
    pdf.set_text_color(30, 30, 30)
    pdf.set_font("Helvetica", "", 8.5)
    fr = data["tiles"].loc[data["ok"], "label"].value_counts(normalize=True)
    for c in sorted(CLASSES, key=lambda c: -fr.get(c, 0)):
        y = pdf.get_y()
        pdf.set_fill_color(*hex_rgb(COLORS[c]))
        pdf.rect(x, y + 0.9, 3.2, 3.2, style="F")
        pdf.set_x(x + 5)
        pdf.cell(col_w - 20, 5, latin1(NAMES[c]))
        pdf.cell(15, 5, pct(fr.get(c, 0)), align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_x(x)
    n_bad = int((~data["ok"]).sum())
    if n_bad:
        pdf.set_font("Helvetica", "I", 8)
        pdf.cell(col_w, 5, f"Not analyzed (quality check): {n_bad} squares", new_x=XPos.LEFT, new_y=YPos.NEXT)
    pdf.ln(2)
    pdf.set_x(x)
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.set_text_color(14, 84, 99)
    pdf.cell(col_w, 5, "How sure the AI is", new_x=XPos.LEFT, new_y=YPos.NEXT)
    pdf.set_text_color(30, 30, 30)
    pdf.set_font("Helvetica", "", 8.5)
    n_ok = max(1, int(data["ok"].sum()))
    for level in ("Sure", "Somewhat sure", "Unsure"):
        pdf.cell(col_w, 4.6, f"{level}: {(data['sure'] == level).sum() / n_ok:.0%} of squares",
                 new_x=XPos.LEFT, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "I", 7.5)
    pdf.multi_cell(col_w, 3.6, "Unsure = top score under 60% or CONCH and MUSK disagree.",
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_y(max(top + map_h, pdf.get_y()) + 4)

    # the caveats always fit: they get a fixed block at the bottom of the page
    caveat_h = 6 + 3.6 * len(CAVEATS) + 2
    caveat_top = 284 - caveat_h
    pdf.BOTTOM = caveat_top - 2

    # text sections, most important first (they stop early rather than overflow)
    flags = s.get("high_risk_flags") or []
    tumor = explain.get(s, "tumor.fraction_of_tissue")
    if flags:
        pdf.section("Top risk flags (places to check, not findings)", flags)
    elif tumor != 0:
        pdf.section("Top risk flags", ["None flagged by the map. Vessel and nerve invasion still need a look by eye."])

    top_class, frac = explain.most_common_tissue(s)
    findings = ([f"Top AI prediction for the slide: mostly {NAMES[top_class].lower()} ({pct(frac)} of tissue)."]
                if top_class else []) + explain.key_findings(s)
    if summary_text:
        pdf.section("Summary", [summary_text])
    pdf.section("Key findings", findings)

    regions = s.get("risk_regions") or []
    if regions:
        lines = [f"Level {r['level']} ({RISK_LEVELS[r['level']][0]}): {r.get('title', '')}, "
                 f"{r.get('n_tiles', 0)} squares." + (f" Check: {r['suggested_actions'][0]}"
                                                     if r.get("suggested_actions") else "")
                 for r in regions]
        pdf.section("Risk regions (highest level first)", lines, max_lines=3)

    spots = data["spots"]
    if spots:
        lines = [f"#{i}: {SHORT_NAMES[a]} or {SHORT_NAMES[b]}? {sp['n_tiles']} squares, average score "
                 f"{sp['mean_conf']:.0%}" + (f", models disagree on {pct(sp['disagree_share'])}"
                                             if data["has_models"] else "") + "."
                 for i, sp in enumerate(spots, 1) for a, b in [sp["pair"]]]
        pdf.section("Where the AI is unsure (check by eye)", lines, max_lines=5)

    pdf.set_y(caveat_top)
    pdf.heading("Caveats")
    pdf.set_font("Helvetica", "", 7.5)
    for line in CAVEATS:
        pdf.cell(0, 3.6, latin1(f"- {line}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # footer
    pdf.set_xy(12, 286)
    pdf.set_font("Helvetica", "I", 7.5)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 4, latin1(f"{FOOTER} Made with Colon Tumor Front Analyzer (Ashby Prize Hackathon 2026)."
                          + (" DEMO data." if slide.is_demo else "")), align="C")
    return bytes(pdf.output())
