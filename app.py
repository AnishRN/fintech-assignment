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
# Create 6 meaningful plots
# ----------------------------
def create_meaningful_plots(df):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.tight_layout(pad=4)

    # 1. Total Due per Bank
    bank_totals = df.groupby("Bank Name")["Total Amount Due"].sum()
    bank_totals.plot(kind="bar", color="#4F81BD", edgecolor="black", ax=axes[0,0])
    axes[0,0].set_title("Total Amount Due per Bank")
    axes[0,0].set_ylabel("Amount (₹)")
    axes[0,0].grid(axis='y', linestyle='--', alpha=0.7)

    # 2. Share of Total Due by Bank
    bank_totals.plot(kind="pie", autopct='%1.1f%%', startangle=140, ax=axes[0,1], colors=plt.cm.Paired.colors)
    axes[0,1].set_ylabel("")
    axes[0,1].set_title("Share of Total Due by Bank")

    # 3. Histogram of Statement Amounts
    df["Total Amount Due"].plot(kind="hist", bins=10, edgecolor='black', color="#FF7F0E", ax=axes[0,2])
    axes[0,2].set_title("Distribution of Statement Amounts")
    axes[0,2].set_xlabel("Amount (₹)")
    axes[0,2].set_ylabel("Count")

    # 4. Confidence vs Amount
    axes[1,0].scatter(df["Total Amount Due"], df["Avg Confidence (%)"], color="#2CA02C", alpha=0.7)
    axes[1,0].set_title("Confidence vs Amount")
    axes[1,0].set_xlabel("Amount (₹)")
    axes[1,0].set_ylabel("Avg Confidence (%)")
    axes[1,0].grid(True, linestyle='--', alpha=0.5)

    # 5. Statements Due per Month
    df['Due Month'] = pd.to_datetime(df['Payment Due Date'], errors='coerce').dt.to_period('M')
    due_month_counts = df.groupby('Due Month').size()
    due_month_counts.plot(kind="bar", color="#D62728", edgecolor='black', ax=axes[1,1])
    axes[1,1].set_title("Number of Statements Due per Month")
    axes[1,1].set_xlabel("Month")
    axes[1,1].set_ylabel("Count")
    axes[1,1].tick_params(axis='x', rotation=45)

    # 6. Avg Amount per Card
    avg_card = df.groupby("Card Last 4")["Total Amount Due"].mean()
    avg_card.plot(kind="bar", color="#9467BD", edgecolor='black', ax=axes[1,2])
    axes[1,2].set_title("Average Amount Due per Card")
    axes[1,2].set_ylabel("Amount (₹)")
    axes[1,2].set_xlabel("Card Last 4")
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

    full_text = ""  # for user Q&A
    for pdf_file in uploaded_files:
        with st.spinner(f"Processing {pdf_file.name}..."):
            pdf_text = extract_text_from_pdf(pdf_file)
            full_text += pdf_text + " "
            extracted_data = extract_fields_with_qa(pdf_text, qa_pipeline)
            extracted_data["file_name"] = pdf_file.name
            cleaned_data = clean_extracted_data(extracted_data)
            all_extracted_data.append(cleaned_data)

    df = pd.DataFrame(all_extracted_data)
    df["Total Amount Due"] = df["Total Amount Due"].astype(float)
    df["Avg Confidence (%)"] = df["Avg Confidence (%)"].astype(float)

    # Display table
    st.subheader("📊 Extracted Information")
    st.dataframe(df.style.format({"Total Amount Due": "₹{:,.2f}"}))

    # Summary
    total_due_sum = df["Total Amount Due"].sum()
    avg_confidence = df["Avg Confidence (%)"].mean()
    summary_text = f"Processed {len(df)} statements. Total due: ₹{total_due_sum:,.2f}. Average confidence: {avg_confidence:.2f}%."
    st.subheader("📈 Summary Insights")
    st.markdown(summary_text)

    # Plots
    fig = create_meaningful_plots(df)
    st.pyplot(fig)

    # Downloads
    csv_file = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Extracted Data as CSV", csv_file, "credit_statements_data.csv", "text/csv")
    pdf_buffer = generate_pdf(df, summary_text, fig)
    st.download_button("📄 Download Full Report as PDF", pdf_buffer, "credit_statements_report.pdf", "application/pdf")

    # ----------------------------
    # User Q&A Section
    # ----------------------------
    st.subheader("❓ Ask Questions About Your Statements")
    user_question = st.text_input("Type your question here (e.g., 'What is the total due for Bank X?')")

    if user_question:
        if full_text:
            with st.spinner("Finding answer..."):
                try:
                    answer = qa_pipeline(question=user_question, context=full_text)
                    st.success(f"**Answer:** {answer.get('answer', 'Not found')}")
                    st.info(f"Confidence: {round(answer.get('score',0)*100,2)}%")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please upload PDF statements first.")
