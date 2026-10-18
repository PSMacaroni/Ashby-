"""
Exports for one slide, made when someone clicks a download button:
  - map_png:    a labeled picture of the map (title, map, color key), e.g. for the slide deck
  - pdf_report: a one-page PDF report: map, tissue mix, flags to check, findings, risk regions,
                unsure areas, and every caveat from the analysis
Both use a white page (for printing) and only restate what the analysis says.
"""
import io
from datetime import date

from PIL import Image, ImageDraw, ImageFont

import explain
import slide_data
import tile_map
from common import CLASS_COLORS, CLASSES, NAMES, get, pct

FOOTER = "Decision support only, not a diagnosis."


# ---------------------------------------------------------------- map picture

def map_picture(slide, mode="tissue", opacity=0.55, focus=None):
    """The map with a title above and a color key below, as a PIL image."""
    data = slide_data.load(slide)
    img = tile_map.map_image(slide, mode, opacity, True, focus)
    if img.width < 1400:                        # make small thumbnails bigger, keeping tile edges crisp
        img = img.resize((img.width * 2, img.height * 2), Image.NEAREST)

    pad, width = 28, img.width + 56
    title_font, font, small = (ImageFont.load_default(size=s) for s in (30, 21, 18))
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    rows, row, x = [], [], pad                  # lay out the color key in rows that fit
    for label, color in tile_map.legend_items(data, mode):
        w = 24 + 8 + probe.textlength(label, font=font) + 28
        if row and x + w > width - pad:
            rows.append(row)
            row, x = [], pad
        row.append((label, color, x))
        x += w
    if row:
        rows.append(row)

    header, key_h, footer = 88, 36 * len(rows) + 14, 42
    canvas = Image.new("RGB", (width, header + img.height + key_h + footer), "white")
    d = ImageDraw.Draw(canvas)
    d.text((pad, 16), f"{slide.name} - {tile_map.MODES.get(mode, mode)}", fill=(20, 20, 20), font=title_font)
    d.text((pad, 56), f"Case {slide.case_id} | Slide {slide.slide_id} | {get(data.summary, 'model.name', '')}",
           fill=(95, 95, 95), font=small)
    canvas.paste(img, (pad, header))
    y = header + img.height + 12
    for row in rows:
        for label, color, x in row:
            d.rounded_rectangle([x, y + 3, x + 24, y + 27], radius=5, fill=tile_map.hex_rgb(color))
            d.text((x + 32, y + 3), label, fill=(30, 30, 30), font=font)
        y += 36
    d.text((pad, canvas.height - footer + 10), FOOTER + (" Demo slide: fake data." if slide.is_demo else "")
           + f" Tumor Front Analyzer, {date.today():%Y-%m-%d}.", fill=(110, 110, 110), font=small)
    return canvas


def map_png(slide, mode="tissue", opacity=0.55, focus=None):
    buf = io.BytesIO()
    map_picture(slide, mode, opacity, focus).save(buf, "PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- one-page PDF report

# The PDF's built-in fonts only know Latin-1 characters, so swap out the few others we use
SWAPS = {"→": "->", "—": "-", "–": "-", "≥": ">=", "≤": "<=", "α": "alpha", "⚠": "!", "‘": "'", "’": "'",
         "“": '"', "”": '"', "…": "..."}
TEAL = (14, 84, 99)


def latin1(text):
    text = str(text)
    for a, b in SWAPS.items():
        text = text.replace(a, b)
    return text.encode("latin-1", "replace").decode("latin-1")


def pdf_report(slide, summary_text=None):
    """A one-page PDF for a slide. summary_text: optional paragraph (e.g. Lumen's) shown first."""
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    data = slide_data.load(slide)
    s, g = data.summary, data.geom
    pdf = FPDF("P", "mm", "A4")
    pdf.set_auto_page_break(False)
    pdf.set_margins(12, 12, 12)
    pdf.add_page()
    nl = dict(new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def heading(text):
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*TEAL)
        pdf.cell(0, 6, latin1(text), **nl)
        pdf.set_text_color(30, 30, 30)

    def lines(items, bottom, size=8.3, max_items=None):
        pdf.set_font("Helvetica", "", size)
        for n, text in enumerate(items):
            if pdf.get_y() > bottom - 5 or (max_items and n >= max_items):
                pdf.set_font("Helvetica", "I", size)
                pdf.cell(0, 4, f"+ {len(items) - n} more in the app", **nl)
                break
            pdf.set_x(13.5)
            pdf.multi_cell(pdf.epw - 1.5, 4.1, latin1(f"- {text}"), **nl)
        pdf.ln(1.6)

    # header band + title
    pdf.set_fill_color(11, 13, 16)
    pdf.rect(0, 0, 210, 15, style="F")
    pdf.set_xy(12, 4)
    pdf.set_font("Helvetica", "B", 11.5)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(130, 7, "Tumor Front Analyzer  |  slide report")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 7, f"{date.today():%Y-%m-%d}", align="R")
    pdf.set_xy(12, 19)
    pdf.set_text_color(20, 20, 20)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 8, latin1(slide.name), **nl)
    pdf.set_font("Helvetica", "", 8.3)
    pdf.set_text_color(95, 95, 95)
    pdf.multi_cell(0, 4.1, latin1(f"Case {slide.case_id}  |  Slide {slide.slide_id}  |  Entered {slide.entered}  |  "
                                  f"{int(data.ok.sum()):,} squares of {g['tile_um']:.0f} µm  |  "
                                  f"Model: {get(s, 'model.name', 'not recorded')}"), **nl)
    if slide.is_demo:
        pdf.ln(1)
        pdf.set_fill_color(255, 236, 179)
        pdf.set_text_color(90, 70, 0)
        pdf.set_font("Helvetica", "B", 8.3)
        pdf.cell(0, 5.5, "  Demo slide: made-up data used to build the app. Every number here is fake.", fill=True, **nl)
    pdf.ln(2.5)

    # map (left) + tissue mix (right)
    top = pdf.get_y()
    img = tile_map.map_image(slide, "tissue", 0.55)
    map_h = min(80, 120 * img.height / img.width)
    map_w = map_h * img.width / img.height
    jpg = io.BytesIO()
    img.save(jpg, "JPEG", quality=90)          # JPEG keeps the PDF small
    pdf.image(jpg, x=12, y=top, w=map_w, h=map_h)
    pdf.set_draw_color(200, 200, 200)
    pdf.rect(12, top, map_w, map_h)
    x = 12 + map_w + 6
    col_w = 198 - x
    pdf.set_xy(x, top)
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.set_text_color(*TEAL)
    pdf.cell(col_w, 5, "Tissue mix", new_x=XPos.LEFT, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 8.3)
    pdf.set_text_color(30, 30, 30)
    fr = s.get("tissue_fractions") or {}
    for c in sorted(CLASSES, key=lambda c: -(fr.get(c) or 0)):
        y = pdf.get_y()
        pdf.set_fill_color(*tile_map.hex_rgb(CLASS_COLORS[c]))
        pdf.rect(x, y + 0.9, 3.2, 3.2, style="F")
        pdf.set_x(x + 5)
        pdf.cell(col_w - 20, 5, latin1(NAMES[c]))
        pdf.cell(15, 5, pct(fr.get(c, 0)), align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_x(x)
    pdf.ln(2)
    pdf.set_x(x)
    pdf.set_font("Helvetica", "I", 7.8)
    pdf.multi_cell(col_w, 3.7, latin1(get(s, "uncertainty.text", "")), **nl)
    pdf.set_y(max(top + map_h, pdf.get_y()) + 4)

    # caveats get a fixed block at the bottom, so they always fit
    caveats = s.get("caveats") or []
    pdf.set_font("Helvetica", "", 7.2)
    n_lines = sum(1 + int(pdf.get_string_width(latin1("- " + c)) // (pdf.epw - 2)) for c in caveats)
    caveat_top = 283 - (6 + 3.4 * n_lines)

    flags = s.get("high_risk_flags") or []
    if summary_text:
        heading("Summary")
        lines([summary_text], caveat_top)
    if flags:
        heading("To check (places to look, not findings)")
        lines(flags, caveat_top)
    heading("Findings")
    lines([f"{k}: {v}" for k, v in explain.findings(s)], caveat_top)
    cards = get(s, "risk.cards", []) or []
    if cards and pdf.get_y() < caveat_top - 14:
        heading("Risk regions (highest level first)")
        lines([f"Level {c['level']}, {c['n_tiles']} squares: " + "; ".join(c.get("features", [])) for c in cards],
              caveat_top, max_items=3)

    pdf.set_y(caveat_top)
    heading("Caveats")
    pdf.set_font("Helvetica", "", 7.2)
    for c in caveats:
        pdf.multi_cell(0, 3.4, latin1(f"- {c}"), **nl)

    pdf.set_xy(12, 286)
    pdf.set_font("Helvetica", "I", 7.5)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 4, latin1(f"{FOOTER} Tumor Front Analyzer (Ashby Prize Hackathon 2026)."
                          + (" Demo data." if slide.is_demo else "")), align="C")
    return bytes(pdf.output())
