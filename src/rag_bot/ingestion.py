from pypdf import PdfReader
from langchain_core.documents import Document

def ingest_pdf(pdf_path: str):
    reader = PdfReader(pdf_path)
    documents = []
    for page_number, page in enumerate(reader.pages):
        text = page.extract_text() or ""

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": pdf_path,
                    "page": page_number
                }
            )
        )
    print(f"Total pages loaded: {len(documents)}")

    for doc in documents:
        print("Page:", doc.metadata["page"])
        print(doc.page_content[:200])

    return documents


if __name__ == "__main__":
    docs = ingest_pdf(
        "Build-a-large-language-models.pdf"
    )