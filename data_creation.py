from langchain_community.document_loaders import PyMuPDFLoader
import fitz
def text_loader(file_path):
    doc = fitz.open(file_path)
    pages_content = []
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text = page.get_text()
        pages_content.append(text)
    doc.close()
    return pages_content