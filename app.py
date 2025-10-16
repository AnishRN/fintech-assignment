import streamlit as st
from PyPDF2 import PdfReader
import re
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO

# PDF generation imports
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# ----------------------------
# Helper functions
# ----------------------------
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_resource
def load_qa_pipeline():
    model_path = "yakul259/credit-statement-scraper"  # replace with your model
    return pipeline("question-answering", model=model_path, tokenizer=model_path)

def extract_fields_with_qa(text, qa_pipeline):
    questions = {
        "bank_name": "Which bank issued this credit card statement?",
        "card_last4": "What are the last 4 digits of the credit card?",
        "billing_cycle": "What is the billing cycle or statement period?",
        "payment_due_date": "What is the payment due date?",
        "total_amount_due": "What is the total amount due?"
    }

    answers = {}
    for key, question in questions.items():
        try:
            result = qa_pipeline(question=question, context=text)
            answers[key] = {
                "answer": result.get("answer", "Not found"),
                "score": round(result.get("score", 0) * 100, 2)
            }
        except Exception:
            answers[key] = {"answer": "Not found", "score": 0.0}
    return answers

def clean_text(s):
    if not s:
        return "Not found"
    return re.sub(r'\s+', ' ', str(s)).strip()

def normalize_amount(amount):
    if not amount:
        return 0.0
    amount = amount.replace('₹','').replace('$','').replace(',','').strip()
    match = re.search(r'[\d\.]+', amount)
    return float(match.group(0)) if match else 0.0

def normalize_date(date_str):
    return clean_text(date_str)

def clean_extracted_data(data):
    numeric_scores = [
        v["score"] for v in data.values()
        if isinstance(v, dict) and "score" in v
    ]
    avg_conf = round(sum(numeric_scores) / len(numeric_scores), 2) if numeric_scores else 0

    return {
        "File Name": data.get("file_name", ""),
        "Bank Name": clean_text(data["bank_name"]["answer"]),
        "Card Last 4": clean_text(data["card_last4"]["answer"]),
        "Billing Cycle": clean_text(data["billing_cycle"]["answer"]),
        "Payment Due Date": normalize_date(data["payment_due_date"]["answer"]),
        "Total Amount Due": normalize_amount(data["total_amount_due"]["answer"]),
        "Avg Confidence (%)": avg_conf
    }

# ----------------------------
# Create meaningful insight plots
# ----------------------------
def create_insightful_plots(df):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.tight_layout(pad=4)

    # 1. Total Amount Due per Bank
    bank_totals = df.groupby("Bank Name")["Total Amount Due"].sum().sort_values(ascending=False)
    bank_totals.plot(kind="bar", color="#4F81BD", edgecolor="black", ax=axes[0,0])
    axes[0,0].set_title("Total Amount Due per Bank")
    axes[0,0].set_ylabel("Amount (₹)")
    axes[0,0].grid(axis='y', linestyle='--', alpha=0.7)

    # 2. Average Payment Amount per Card
    card_avg = df.groupby("Card Last 4")["Total Amount Due"].mean().sort_values(ascending=False)
    card_avg.plot(kind="bar", color="#FF7F0E", edgecolor="black", ax=axes[0,1])
    axes[0,1].set_title("Average Amount per Card")
    axes[0,1].set_ylabel("Average Amount (₹)")
    axes[0,1].set_xlabel("Card Last 4")
    axes[0,1].tick_params(axis='x', rotation=45)

    # 3. Payment Due Date Distribution
    df['Due Day'] = pd.to_datetime(df['Payment Due Date'], errors='coerce').dt.day
    df['Due Day'].plot(kind='hist', bins=31, rwidth=0.8, color="#2CA02C", ax=axes[0,2])
    axes[0,2].set_title("Payment Due Date Distribution")
    axes[0,2].set_xlabel("Day of Month")
    axes[0,2].set_ylabel("Number of Statements")

    # 4. High-Value Statements per Bank
    threshold = 50000
    high_value_counts = df[df['Total Amount Due'] > threshold].groupby("Bank Name").size().sort_values()
    high_value_counts.plot(kind='barh', color="#D62728", edgecolor="black", ax=axes[1,0])
    axes[1,0].set_title(f"High-Value Statements per Bank (>₹{threshold})")
    axes[1,0].set_xlabel("Number of Statements")
    axes[1,0].set_ylabel("Bank Name")

    # 5. Top 5 Cards by Average Amount
    top5_cards = df.groupby("Card Last 4")["Total Amount Due"].mean().sort_values(ascending=False).head(5)
    top5_cards.plot(kind="bar", color="#17BECF", edgecolor="black", ax=axes[1,1])
    axes[1,1].set_title("Top 5 Cards by Average Amount")
    axes[1,1].set_ylabel("Average Amount (₹)")
    axes[1,1].set_xlabel("Card Last 4")
    axes[1,1].tick_params(axis='x', rotation=45)

    # 6. Number of Statements per Bank
    statements_per_bank = df.groupby("Bank Name").size().sort_values(ascending=False)
    statements_per_bank.plot(kind="bar", color="#BCBD22", edgecolor="black", ax=axes[1,2])
    axes[1,2].set_title("Number of Statements per Bank")
    axes[1,2].set_ylabel("Count")
    axes[1,2].set_xlabel("Bank Name")
    axes[1,2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    return fig


# ----------------------------
# PDF Generator
# ----------------------------
def generate_pdf(df, summary_text, fig):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    elements = []
    styles = getSampleStyleSheet()

    title = Paragraph("Credit Card Statement Extraction Report", styles['Title'])
    date_text = Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal'])
    summary = Paragraph(summary_text, styles['Normal'])

    elements += [title, Spacer(1, 10), date_text, Spacer(1, 10), summary, Spacer(1, 20)]

    # Table
    data = [df.columns.tolist()] + df.values.tolist()
    wrapped_data = []
    for row in data:
        wrapped_row = [Paragraph(str(cell), styles['Normal']) for cell in row]
        wrapped_data.append(wrapped_row)

    total_width = 10.5 * 72
    col_width = total_width / len(df.columns)
    col_widths = [col_width]*len(df.columns)

    table = Table(wrapped_data, colWidths=col_widths, hAlign='CENTER')
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#4F81BD")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("BOTTOMPADDING", (0,0), (-1,0), 10),
        ("BACKGROUND", (0,1), (-1,-1), colors.whitesmoke),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey)
    ]))
    elements.append(table)
    elements.append(Spacer(1, 20))

    # Save figure as image and add to PDF
    fig_buffer = BytesIO()
    fig.savefig(fig_buffer, format='PNG', dpi=150, bbox_inches='tight')
    fig_buffer.seek(0)
    img = Image(fig_buffer)
    img.drawHeight = 400
    img.drawWidth = 720
    elements.append(img)

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Credit Card Statement Extractor", page_icon="💳", layout="wide")
st.title("💳 Credit Card Statement Extractor")

uploaded_files = st.file_uploader(
    "Upload one or more credit card statement PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    qa_pipeline = load_qa_pipeline()
    all_extracted_data = []

    for pdf_file in uploaded_files:
        with st.spinner(f"Processing {pdf_file.name}..."):
            pdf_text = extract_text_from_pdf(pdf_file)
            extracted_data = extract_fields_with_qa(pdf_text, qa_pipeline)
            extracted_data["file_name"] = pdf_file.name
            cleaned_data = clean_extracted_data(extracted_data)
            all_extracted_data.append(cleaned_data)

    df = pd.DataFrame(all_extracted_data)

    st.subheader("Extracted Information for All PDFs")
    st.dataframe(df.style.format({"Total Amount Due": "₹{:.2f}", "Avg Confidence (%)": "{:.2f}%"}))

    # Download CSV
    csv_file = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download CSV",
        data=csv_file,
        file_name="credit_statements_data.csv",
        mime="text/csv"
    )

    # Generate and display plots
    fig = create_meaningful_insight_plots(df)
    st.subheader("Insights from Data")
    st.pyplot(fig)

    # Generate PDF
    summary_text = f"Processed {len(df)} statements from {len(df['Bank Name'].unique())} banks."
    pdf_buffer = generate_pdf(df, summary_text, fig)
    st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf_buffer,
        file_name="credit_statements_report.pdf",
        mime="application/pdf"
    )

# ----------------------------
# Simple Chatbot-style Q&A
# ----------------------------
st.subheader("💬 Ask Questions About the Uploaded Statements")
user_question = st.text_input("Type your question here:")

if user_question and uploaded_files:
    qa_pipeline = load_qa_pipeline()
    combined_text = " ".join(extract_text_from_pdf(f) for f in uploaded_files)
    try:
        result = qa_pipeline(question=user_question, context=combined_text)
        st.write(f"**Answer:** {result.get('answer', 'Not found')}")
        st.write(f"*Confidence:* {round(result.get('score',0)*100,2)}%")
    except Exception as e:
        st.error(f"Error processing question: {e}")

