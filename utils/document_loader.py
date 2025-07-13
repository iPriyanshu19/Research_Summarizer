import PyPDF2

def load_pdf(file):
    pdf = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

def load_txt(file):
    return file.read().decode("utf-8")
