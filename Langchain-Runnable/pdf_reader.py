from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace
)
from langchain_community.vectorstores import FAISS

load_dotenv()


# --------------------------------------------------
# 1. Load document
# --------------------------------------------------

loader = TextLoader("docs.txt")

documents = loader.load()


# --------------------------------------------------
# 2. Split document into chunks
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)


# --------------------------------------------------
# 3. Create Hugging Face Embeddings
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 4. Store embeddings in FAISS
# --------------------------------------------------

vectorstore = FAISS.from_documents(
    docs,
    embeddings
)


# --------------------------------------------------
# 5. Create Retriever
# --------------------------------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# --------------------------------------------------
# 6. User Query
# --------------------------------------------------

query = "What are the key takeaways from the documents?"


# --------------------------------------------------
# 7. Retrieve relevant documents
# --------------------------------------------------

retrieved_docs = retriever.invoke(query)


# --------------------------------------------------
# 8. Combine retrieved documents
# --------------------------------------------------

retrieved_text = "\n\n".join(
    doc.page_content
    for doc in retrieved_docs
)


# --------------------------------------------------
# 9. Hugging Face LLM
# --------------------------------------------------

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.2,
    max_new_tokens=300
)

model = ChatHuggingFace(llm=llm)


# --------------------------------------------------
# 10. Create Prompt
# --------------------------------------------------

prompt = f"""
You are a helpful assistant.

Answer the question using ONLY the provided context.

Context:
{retrieved_text}

Question:
{query}

Answer:
"""


# --------------------------------------------------
# 11. Generate Answer
# --------------------------------------------------

response = model.invoke(prompt)


print(response.content)