# 💳 Credit Card Statement Extractor

This project provides a **Streamlit web application** that extracts structured information from **credit card statement PDFs** using a **question-answering (QA) pipeline** powered by a pre-trained Hugging Face model.  

The app can process **single or multiple PDFs** and output structured data including bank name, last 4 digits of the card, billing cycle, payment due date, and total amount due. Users can download the extracted data as a CSV.

---

## 🚀 Live App

Try the app online:  
[Streamlit Community Cloud](https://fintech-assignment-bjpete769nwndlniwpnyb5.streamlit.app/)  

Or view the model on Hugging Face Spaces:  
[Hugging Face Space](https://huggingface.co/yakul259/credit-statement-scraper)

---

## 📝 Features

- Upload **one or more PDF credit card statements**.
- Extract key fields:
  - Bank Name
  - Card Last 4 Digits
  - Billing Cycle / Statement Period
  - Payment Due Date
  - Total Amount Due
- Display extracted information interactively in a table.
- Download all extracted data as a CSV file.
- Works with **pre-trained QA models** from Hugging Face.
- Clean and normalized outputs (dates and amounts).

---

## 📁 Project Structure
```
credit-card-statement-extractor/
│
├─ app.py # Streamlit main app
├─ requirements.txt # Python dependencies
├─ model/ # Saved Hugging Face QA model (optional if using HF repo)
├─ README.md # Project documentation
└─ sample_pdfs/ # Example PDF statements (optional)
```
---

## 🛠 How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/your-username/credit-card-statement-extractor.git
cd credit-card-statement-extractor
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py

```

##⚙️ Requirements

Python 3.12+ (recommended for Streamlit Cloud compatibility)
Streamlit
PyPDF2
pandas
transformers
torch

##📂 Model Options
Use a Hugging Face model hub path, e.g., "username/credit-statement-scraper".

#⚠️ Disclaimer
This tool is intended for educational and personal use only.
Do not rely on this application for financial, legal, or commercial decisions.

##👨‍💻 Author
### App:
Name: Anish
GitHub: https://github.com/AnishRN

---

