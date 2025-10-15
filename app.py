import streamlit as st
from PyPDF2 import PdfReader
import re
from transformers import pipeline
import pandas as pd

# ----------------------------
# Helper functions
# ----------------------------
def extract_text_from_pdf(pdf_file):
    """Extracts and cleans text from a PDF file."""
    reader = PdfReader(pdf_file)
    text = " ".join(page.extract_text() for page in reader.pages)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_resource
def load_qa_pipeline():
    """Loads the Hugging Face QA pipeline from a saved model."""
    model_path = "yakul259/credit-statement-scraper"  # replace with your model
    return pipeline("question-answering", model=model_path, tokenizer=model_path)

def extract_fields_with_qa(text, qa_pipeline):
    """Uses the QA pipeline to extract key fields from the text."""
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
            answers[key] = result.get("answer", "Not found")
        except:
            answers[key] = "Not found"
    return answers

def clean_text(s):
    if not s:
        return "Not found"
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def normalize_amount(amount):
    if not amount:
        return "0"
    amount = amount.replace('₹','').replace('$','').replace(',','').strip()
    match = re.search(r'[\d\.]+', amount)
    return match.group(0) if match else "0"

def normalize_date(date_str):
    return clean_text(date_str)

def clean_extracted_data(data):
    """Cleans and standardizes extracted fields."""
    return {
        "File Name": data.get("file_name", ""),
        "Bank Name": clean_text(data.get("bank_name","")),
        "Card Last 4": clean_text(data.get("card_last4","")),
        "Billing Cycle": clean_text(data.get("billing_cycle","")),
        "Payment Due Date": normalize_date(data.get("payment_due_date","")),
        "Total Amount Due": normalize_amount(data.get("total_amount_due",""))
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

    # Display in a dataframe
    st.subheader("Extracted Information for All PDFs")
    df = pd.DataFrame(all_extracted_data)
    st.dataframe(df.style.format({"Total Amount Due": "${}"}))

    # Download combined CSV
    csv_file = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download All Extracted Data as CSV",
        data=csv_file,
        file_name="all_credit_statements_data.csv",
        mime="text/csv",
    )
