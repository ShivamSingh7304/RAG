from rag_bot.ingestion import ingest_pdf

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
pdf_path="Build-a-large-language-models.pdf"
def vector_stores():
    documents =ingest_pdf(pdf_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )

    chunks=splitter.split_documents(documents)
    print(f"length of chunks created{len(chunks)}")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    vector_store.save_local("vector_store/faiss_index")
     
    return vector_store

if __name__ == "__main__":
    vector_stores()