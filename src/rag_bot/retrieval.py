from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

embedding_model= HuggingFaceEmbeddings(
     model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.load_local(
    "vector_store/faiss_index",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(
    search_kwargs={"k":4}
)

Model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)



prompt =PromptTemplate.from_template(
    """
You are an expert software engineering assistant.

Answer the user's question using the provided context.

If the answer is not present in the context, reply:

"use your knowledge but at last of your response print "
if retrieved context is used print response and then print [rag used]
try to answer using context firstly i not present knowledge then only answer by your kknowledge
if you generate answer which is not present in context then after resonse print [llm used]

Context:
{context}

Question:
{question}

Answer:
"""
)


parser = StrOutputParser()

chain = prompt|Model|parser

def ask_question(question: str):

    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    answer = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    return answer
