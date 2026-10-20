"""
Exports for one slide, made when someone clicks a download button:
  - map_png:    a labeled picture of the map (title, map, color key), e.g. for the slide deck
  - pdf_report: a PDF report: map, tissue mix, high-risk features, key findings, places to check
                (with review state), the pathologist's notes, the full report and every caveat
Both use a white page (for printing) and only restate what the analysis says.
"""
import io
from datetime import date

from PIL import Image, ImageDraw, ImageFont

import report
import slide_data
import tile_map
import worklist
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
    d.text((pad, 22), f"{slide.name}  ·  {tile_map.MODES.get(mode, mode)} map", fill=(20, 20, 20), font=title_font)
    d.text((pad, 58), f"Case {slide.case_id}  ·  {int(data.ok.sum()):,} squares of {data.geom['tile_um']:.0f} µm  ·  "
                      f"{get(data.summary, 'model.name', '')}", fill=(110, 110, 110), font=small)
    canvas.paste(img, (pad, header))
    y = header + img.height + 12
    for row in rows:
        for label, color, x in row:
            d.rounded_rectangle([x, y + 4, x + 24, y + 28], radius=5, fill=tile_map.hex_rgb(color), outline=(200, 200, 200))
            d.text((x + 32, y + 4), label, fill=(40, 40, 40), font=font)
        y += 36
    d.text((pad, canvas.height - 32), f"{FOOTER}  Tumor Front Analyzer, {date.today():%Y-%m-%d}.",
           fill=(130, 130, 130), font=small)
    return canvas


def map_png(slide, mode="tissue", opacity=0.55, focus=None):
    buf = io.BytesIO()
    map_picture(slide, mode, opacity, focus).save(buf, "PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- PDF report

# The PDF's built-in fonts only know Latin-1 characters, so swap out the few others we use
SWAPS = {"→": "->", "—": "-", "–": "-", "≥": ">=", "≤": "<=", "α": "alpha", "⚠": "!", "‘": "'", "’": "'",
         "“": '"', "”": '"', "…": "...", "·": "-", "²": "2", "±": "+/-", "›": ">"}
BLUE = (31, 95, 139)
INK = (27, 31, 36)
GREY = (107, 116, 128)


def latin1(text):
    text = str(text)
    for a, b in SWAPS.items():
        text = text.replace(a, b)
    return text.encode("latin-1", "replace").decode("latin-1")


def pdf_report(slide, summary_text=None):
    """The PDF report for a slide. summary_text: optional paragraph (e.g. Lumen's) shown first."""
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    data = slide_data.load(slide)
    s, g = data.summary, data.geom
    items = worklist.build(s)
    done = slide_data.reviewed(slide.key)
    notes = slide_data.notes(slide.key).strip()

    pdf = FPDF("P", "mm", "A4")
    pdf.set_auto_page_break(True, margin=14)
    pdf.set_margins(12, 12, 12)
    nl = dict(new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def heading(text, size=10.5):
        pdf.ln(1.5)
        pdf.set_font("Helvetica", "B", size)
        pdf.set_text_color(*BLUE)
        pdf.cell(0, 6, latin1(text), **nl)
        pdf.set_text_color(*INK)

    def para(text, size=8.6, style="", color=INK):
        pdf.set_font("Helvetica", style, size)
        pdf.set_text_color(*color)
        pdf.multi_cell(0, 4.2, latin1(text), **nl)
        pdf.set_text_color(*INK)

    def bullets(items_, size=8.6):
        pdf.set_font("Helvetica", "", size)
        for text in items_:
            pdf.set_x(13.5)
            pdf.multi_cell(pdf.epw - 1.5, 4.2, latin1(f"- {text}"), **nl)

    def table(rows, size=8.3, key_w=46):
        for row in rows:
            k, v = row[0], row[1]
            strong = len(row) > 2 and row[2]
            y0 = pdf.get_y()
            pdf.set_font("Helvetica", "", size - 0.6)
            pdf.set_text_color(*GREY)
            pdf.set_x(12)
            pdf.multi_cell(key_w, 4.0, latin1(k), new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.set_font("Helvetica", "B" if strong else "", size)
            pdf.set_text_color(*INK)
            pdf.set_xy(12 + key_w, y0)
            pdf.multi_cell(pdf.epw - key_w, 4.0, latin1(v), **nl)
            pdf.set_draw_color(225, 228, 232)
            pdf.line(12, pdf.get_y() + 0.4, 12 + pdf.epw, pdf.get_y() + 0.4)
            pdf.ln(1.2)

    pdf.add_page()
    # header band + title
    pdf.set_fill_color(*BLUE)
    pdf.rect(0, 0, 210, 14, style="F")
    pdf.set_xy(12, 3.5)
    pdf.set_font("Helvetica", "B", 11.5)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(130, 7, "Tumor Front Analyzer  |  slide report")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 7, f"{date.today():%Y-%m-%d}", align="R")
    pdf.set_xy(12, 18)
    pdf.set_text_color(*INK)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 8, latin1(slide.name), **nl)
    para(f"Case {slide.case_id}  |  Slide {slide.slide_id}  |  {slide.site or ''}  |  TCGA report: {slide.stage or 'n/a'}  |  "
         f"Processed {slide.entered}  |  {int(data.ok.sum()):,} squares of {g['tile_um']:.0f} um  |  "
         f"Model: {get(s, 'model.name', 'not recorded')}", size=8.2, color=GREY)
    pdf.ln(1.5)

    # map (left) + tissue mix (right)
    top = pdf.get_y()
    img = tile_map.map_image(slide, "tissue", 0.55)
    map_h = min(78, 120 * img.height / img.width)
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
    pdf.set_text_color(*BLUE)
    pdf.cell(col_w, 5, "Tissue mix", new_x=XPos.LEFT, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 8.3)
    pdf.set_text_color(*INK)
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
    pdf.set_y(max(top + map_h, pdf.get_y()) + 3)

    flags = s.get("high_risk_flags") or []
    if summary_text:
        heading("Summary")
        para(summary_text)
    heading(f"High-risk features ({len(flags)})" if flags else "High-risk features")
    bullets(flags or ["None flagged by the map. Vessel and nerve invasion still need a look by eye."])
    heading("Key findings")
    table(report.key_findings(s))

    if notes:
        heading("Pathologist's notes")
        para(notes)

    if items:
        n_done = sum(1 for it in items if it["id"] in done)
        heading(f"Places to check ({n_done} of {len(items)} marked reviewed in the app)")
        for group, name in worklist.GROUPS:
            mine = [it for it in items if it["group"] == group]
            if not mine:
                continue
            para(name, size=8.8, style="B")
            bullets([f"{'[x]' if it['id'] in done else '[ ]'} {it['label'] + ' ' if it['label'] else ''}{it['title']}"
                     + (f" - {it['sub']}" if it.get("sub") else "")
                     + (f" (row {it['loc'][0]}, col {it['loc'][1]})" if it["loc"] else "") for it in mine], size=8.2)

    pdf.add_page()
    heading("Full report", size=12)
    para("Every number the analysis wrote to summary.json, in report order. Places are given as map squares (row, column).",
         size=8, style="I", color=GREY)
    for sec in report.sections(s):
        if sec.key == "caveats":
            continue
        heading(sec.title)
        if sec.rows:
            table(sec.rows)
        if sec.items:
            bullets([f"{it['title']}" + (f" - {it['sub']}" if it.get("sub") else "")
                     + (f" (row {it['loc'][0]}, col {it['loc'][1]})" if it.get("loc") else "") for it in sec.items], size=8.2)
        if sec.note:
            para(sec.note, size=7.8, style="I", color=GREY)

    heading("Caveats (read before using any number)")
    bullets(s.get("caveats") or [], size=7.8)
    pdf.ln(3)
    para(f"{FOOTER} Tumor Front Analyzer (Ashby Prize Hackathon 2026). TCGA slides are public, de-identified data.",
         size=7.5, style="I", color=GREY)
    return bytes(pdf.output())
