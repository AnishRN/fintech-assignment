import streamlit as st
from PyPDF2 import PdfReader
import re
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt

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
    model_path = "yakul259/credit-statement-scraper"
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
    s = re.sub(r'\s+', ' ', str(s)).strip()
    return s

def normalize_amount(amount):
    if not amount:
        return "0"
    amount = amount.replace('₹', '').replace('$', '').replace(',', '').strip()
    match = re.search(r'[\d\.]+', amount)
    return match.group(0) if match else "0"

def normalize_date(date_str):
    return clean_text(date_str)

def clean_extracted_data(data):
    """Cleans and standardizes extracted fields."""
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

    st.subheader("📊 Extracted Information")
    st.dataframe(df.style.format({"Total Amount Due": "₹{}"}))

    # ---- Summary Section ----
    st.subheader("📈 Summary Insights")
    total_due_sum = df["Total Amount Due"].astype(float).sum()
    avg_confidence = df["Avg Confidence (%)"].mean()
    st.markdown(f"""
    **Total Statements:** {len(df)}  
    **Total Due (All):** ₹{total_due_sum:,.2f}  
    **Average Confidence:** {avg_confidence:.2f}%
    """)

    # ---- Chart ----
    fig, ax = plt.subplots()
    df.groupby("Bank Name")["Total Amount Due"].apply(lambda x: x.astype(float)).sum().plot(kind="bar", ax=ax)
    ax.set_ylabel("Total Amount Due (₹)")
    ax.set_title("Total Amount Due per Bank")
    st.pyplot(fig)

    # ---- Download Buttons ----
    csv_file = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Extracted Data as CSV",
        data=csv_file,
        file_name="credit_statements_data.csv",
        mime="text/csv"
    )

    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    import io

    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = [Paragraph("Credit Card Statement Extraction Report", styles["Heading1"]), Spacer(1, 12)]

    table_data = [df.columns.tolist()] + df.values.tolist()
    table = Table(table_data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey)
    ]))

    elements.append(table)
    doc.build(elements)
    pdf_buffer.seek(0)

    st.download_button(
        label="⬇️ Download Extracted Data as PDF",
        data=pdf_buffer,
        file_name="credit_statements_report.pdf",
        mime="application/pdf"
    )
