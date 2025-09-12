import streamlit as st
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import PyPDF2
from docx import Document
import io


def extract_text_from_pdf(file_bytes):
    """Extracts text from a PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None


def extract_text_from_docx(file_bytes):
    """Extracts text from a DOCX file."""
    try:
        doc = Document(io.BytesIO(file_bytes))
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX file: {e}")
        return None


def generate_word_cloud(text):
    """Generates and displays a word cloud from the given text."""
    if not text.strip():
        st.warning("The uploaded file is empty or contains no readable text.")
        return

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="black",
        stopwords=STOPWORDS,
        min_font_size=10,
        collocations=False
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot(fig)


def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(layout="wide", page_title="Word Cloud Generator", page_icon="☁")

    # --- Custom Dark Theme CSS ---
    st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: white;
    }
    /* Upload widget box */
    div[data-testid="stFileUploader"] {
        border: 1px dashed #4b5563;
        padding: 20px;
        border-radius: 10px;
        background-color: #1f2937;
    }
    /* Upload text */
    div[data-testid="stFileUploader"] > div {
        color: #d1d5db;
    }
    /* Browse files button */
    button[data-testid="baseButton-secondary"] {
        background-color: #374151;
        color: white;
        border: 1px solid #4b5563;
    }
    button[data-testid="baseButton-secondary"]:hover {
        background-color: #4b5563;
        border: 1px solid #6b7280;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- UI ---
    st.title("☁ Word Cloud from PDF or Word")
    st.write("Upload a PDF or Word document to generate a word cloud of its content.")

    uploaded_file = st.file_uploader(
        "Drag and drop file here",
        type=["pdf", "docx"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_type = uploaded_file.type
        text_content = ""

        with st.spinner(f"Processing {uploaded_file.name}..."):
            if file_type == "application/pdf":
                text_content = extract_text_from_pdf(file_bytes)
            elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text_content = extract_text_from_docx(file_bytes)

        if text_content:
            st.subheader("Generated Word Cloud")
            generate_word_cloud(text_content)
        else:
            st.error("Could not extract text from the uploaded file. Please try another file.")


if _name_ == "_main_":
    main()
