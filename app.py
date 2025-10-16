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
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
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
# Create meaningful plots
# ----------------------------
def create_meaningful_plots(df):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.tight_layout(pad=4)

    # 1. Total Amount Due per Bank
    bank_totals = df.groupby("Bank Name")["Total Amount Due"].sum().sort_values(ascending=False)
    bank_totals.plot(kind="bar", color="#4F81BD", edgecolor="black", ax=axes[0,0])
    axes[0,0].set_title("Total Amount Due per Bank")
    axes[0,0].set_ylabel("Amount (₹)")
    axes[0,0].grid(axis='y', linestyle='--', alpha=0.7)

    # 2. Monthly Spending Trends
    df['Month'] = pd.to_datetime(df['Payment Due Date'], errors='coerce').dt.to_period('M')
    monthly_total = df.groupby('Month')["Total Amount Due"].sum()
    monthly_total.plot(kind="line", marker='o', color="#FF7F0E", ax=axes[0,1])
    axes[0,1].set_title("Total Amount Due per Month")
    axes[0,1].set_ylabel("Total Amount (₹)")
    axes[0,1].set_xlabel("Month")
    axes[0,1].grid(True, linestyle='--', alpha=0.5)
    axes[0,1].tick_params(axis='x', rotation=45)

    # 3. Payment Timeliness (Early/Mid/Late Month)
    due_days = pd.to_datetime(df['Payment Due Date'], errors='coerce').dt.day
    bins = [0,10,20,31]
    labels = ["Early (1-10)", "Mid (11-20)", "Late (21-31)"]
    df['Due Period'] = pd.cut(due_days, bins=bins, labels=labels, include_lowest=True)
    due_period_counts = df['Due Period'].value_counts().reindex(labels)
    due_period_counts.plot(kind="pie", autopct='%1.1f%%', colors=["#2CA02C","#98DF8A","#D0F0C0"], ax=axes[0,2])
    axes[0,2].set_title("Payment Due Distribution")
    axes[0,2].set_ylabel("")

    # 4. High-Value Statements per Bank (> ₹50,000)
    high_value_threshold = 50000
    high_value_df = df[df['Total Amount Due'] > high_value_threshold]
    axes[1,0].scatter(high_value_df["Bank Name"], high_value_df["Total Amount Due"], color="#D62728", s=100)
    axes[1,0].set_title(f"High-Value Statements per Bank (> ₹{high_value_threshold})")
    axes[1,0].set_xlabel("Bank Name")
    axes[1,0].set_ylabel("Amount (₹)")
    axes[1,0].tick_params(axis='x', rotation=45)

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
def generate_pdf(df, fig):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter),
                            rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    elements = []
    styles = getSampleStyleSheet()

    # --- Page 1: Table ---
    title = Paragraph("Credit Card Statement Extraction Report", styles['Title'])
    date_text = Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal'])
    elements += [title, Spacer(1, 10), date_text, Spacer(1, 20)]

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
    elements.append(PageBreak())

    # --- Page 2: Plots ---
    fig_buffer = BytesIO()
    fig.savefig(fig_buffer, format='PNG', dpi=150, bbox_inches='tight')
    fig_buffer.seek(0)
    img = Image(fig_buffer)
    img.drawHeight = 400
    img.drawWidth = 720
    elements.append(img)
    elements.append(PageBreak())

    # --- Page 3: Useful Info ---
    info_title = Paragraph("📌 Useful Information / Summary", styles['Heading2'])
    elements.append(info_title)
    elements.append(Spacer(1, 10))

    total_due_sum = df["Total Amount Due"].sum()
    avg_confidence = df["Avg Confidence (%)"].mean()
    num_statements = len(df)
    num_banks = df["Bank Name"].nunique()
    high_value_threshold = 50000
    high_value_count = (df["Total Amount Due"] > high_value_threshold).sum()
    bank_totals = df.groupby("Bank Name")["Total Amount Due"].sum()
    top_bank = bank_totals.idxmax() if not bank_totals.empty else "N/A"
    top_card = df.groupby("Card Last 4")["Total Amount Due"].mean().idxmax() if not df.empty else "N/A"

    # Info / Summary
    summary_text = f"""
    Number of statements processed: {num_statements}<br/>
    Number of unique banks: {num_banks}<br/>
    Total Amount Due across all statements: ₹{total_due_sum:,.2f}<br/>
    Average QA confidence: {avg_confidence:.2f}%<br/>
    Number of high-value statements (> ₹{high_value_threshold}): {high_value_count}<br/>
    Bank with highest total due: {top_bank}<br/>
    Card with highest average due: {top_card}<br/>
    """
    elements.append(Paragraph(summary_text, styles['Normal']))
    elements.append(Spacer(1, 15))

    # Classification Criteria
    criteria_title = Paragraph("🗂 Classification Criteria", styles['Heading3'])
    elements.append(criteria_title)
    elements.append(Spacer(1, 5))
    classification_text = """
    - **Early / Mid / Late Month Payment Due Dates**: 
      - Early: Days 1–10 of the month
      - Mid: Days 11–20 of the month
      - Late: Days 21–31 of the month
    - **High-Value Statements**: Any statement with Total Amount Due > ₹50,000
    - **Top 5 Cards by Average Due Amount**: Cards with the 5 highest average Total Amount Due across all statements
    """
    elements.append(Paragraph(classification_text, styles['Normal']))
    elements.append(Spacer(1, 15))

    # Terms & Conditions
    terms_title = Paragraph("📃 Terms & Conditions", styles['Heading3'])
    elements.append(terms_title)
    elements.append(Spacer(1, 5))
    terms_text = """
    1. This report is auto-generated based on uploaded PDFs.<br/>
    2. Verify extracted amounts before making financial decisions.<br/>
    3. High-value statements indicate significant expenditures.<br/>
    4. Ensure secure handling of sensitive card information.<br/>
    """
    elements.append(Paragraph(terms_text, styles['Normal']))

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
    full_text = ""

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
    pdf_buffer = generate_pdf(df, fig)
    st.download_button("📄 Download Full Report as PDF", pdf_buffer, "credit_statements_report.pdf", "application/pdf")

    # Chatbot-style Q&A
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.subheader("💬 Ask Your Statements (Chatbot Mode)")
    user_question = st.text_input("Type your question here and press Enter:")

    if user_question and full_text:
        with st.spinner("Thinking..."):
            try:
                answer = qa_pipeline(question=user_question, context=full_text)
                response_text = f"{answer.get('answer','Not found')} (Confidence: {round(answer.get('score',0)*100,2)}%)"
                st.session_state.chat_history.append({"user": user_question, "bot": response_text})
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Display chat history (latest on top)
    for chat in st.session_state.chat_history[::-1]:
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")

