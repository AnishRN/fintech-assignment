from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from io import BytesIO

def generate_pdf(dataframe):
    """Generates a properly formatted and scaled PDF from the extracted dataframe."""
    buffer = BytesIO()

    # Use landscape orientation for better fit
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    elements = []

    styles = getSampleStyleSheet()
    title = Paragraph("Credit Card Statement Extraction Summary", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 12))

    # Convert DataFrame to list of lists for table
    data = [dataframe.columns.tolist()] + dataframe.values.tolist()

    # Wrap long text as Paragraphs
    wrapped_data = []
    for row in data:
        wrapped_row = [Paragraph(str(cell), styles['Normal']) for cell in row]
        wrapped_data.append(wrapped_row)

    # Dynamically calculate column widths
    num_cols = len(dataframe.columns)
    total_width = 10.5 * inch  # usable width in landscape letter
    col_width = total_width / num_cols
    col_widths = [col_width for _ in range(num_cols)]

    # Create the table
    table = Table(wrapped_data, colWidths=col_widths, hAlign='CENTER')

    # Styling
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4F81BD")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
    ]))

    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer
