from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports" / "final_report.md"
OUTPUT = ROOT / "reports" / "final_report.pdf"


def register_fonts():
    candidates = [
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            pdfmetrics.registerFont(TTFont("BodyFont", str(path)))
            return "BodyFont"
    return "Helvetica"


def clean_inline(text):
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"`([^`]+)`", r"<font face='Courier'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text


def make_styles(font_name):
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName=font_name,
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            spaceAfter=14,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportHeading1",
            parent=styles["Heading1"],
            fontName=font_name,
            fontSize=14,
            leading=17,
            spaceBefore=12,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportHeading2",
            parent=styles["Heading2"],
            fontName=font_name,
            fontSize=12,
            leading=15,
            spaceBefore=10,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportBody",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=10.5,
            leading=14,
            alignment=TA_LEFT,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportBullet",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=10.5,
            leading=14,
            leftIndent=14,
            firstLineIndent=-8,
            spaceAfter=3,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CodeBlock",
            parent=styles["Code"],
            fontName="Courier",
            fontSize=8,
            leading=10,
            backColor=colors.whitesmoke,
            borderColor=colors.lightgrey,
            borderWidth=0.5,
            borderPadding=5,
            spaceBefore=5,
            spaceAfter=7,
        )
    )
    return styles


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.drawCentredString(letter[0] / 2.0, 0.45 * inch, str(doc.page))
    canvas.restoreState()


def markdown_to_flowables(markdown, styles):
    story = []
    in_code = False
    code_lines = []
    pending_bullets = []

    def flush_code():
        nonlocal code_lines
        if code_lines:
            story.append(Preformatted("\n".join(code_lines), styles["CodeBlock"]))
            code_lines = []

    def flush_bullets():
        nonlocal pending_bullets
        if pending_bullets:
            items = [
                ListItem(Paragraph(clean_inline(item), styles["ReportBody"]))
                for item in pending_bullets
            ]
            story.append(
                ListFlowable(
                    items,
                    bulletType="bullet",
                    start="circle",
                    leftIndent=18,
                    bulletFontSize=7,
                )
            )
            story.append(Spacer(1, 3))
            pending_bullets = []

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()

        if line.startswith("```"):
            if in_code:
                flush_code()
                in_code = False
            else:
                flush_bullets()
                in_code = True
            continue

        if in_code:
            code_lines.append(line)
            continue

        if not line.strip():
            flush_bullets()
            story.append(Spacer(1, 3))
            continue

        if line.startswith("# "):
            flush_bullets()
            story.append(Paragraph(clean_inline(line[2:]), styles["ReportTitle"]))
            continue

        if line.startswith("## "):
            flush_bullets()
            story.append(Paragraph(clean_inline(line[3:]), styles["ReportHeading1"]))
            continue

        if line.startswith("### "):
            flush_bullets()
            story.append(Paragraph(clean_inline(line[4:]), styles["ReportHeading2"]))
            continue

        if line.startswith("- "):
            pending_bullets.append(line[2:])
            continue

        flush_bullets()
        story.append(Paragraph(clean_inline(line), styles["ReportBody"]))

    flush_code()
    flush_bullets()
    return story


def main():
    font_name = register_fonts()
    styles = make_styles(font_name)
    markdown = SOURCE.read_text(encoding="utf-8")

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title="Reward Design with Generalized Prioritized Sweeping",
    )
    story = markdown_to_flowables(markdown, styles)
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(OUTPUT)


if __name__ == "__main__":
    main()
