import os
from dotenv import load_dotenv

# Text Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings (FREE Local)
from langchain_community.embeddings import HuggingFaceEmbeddings

# Vector Store
from langchain_community.vectorstores import FAISS

# LLM (Paid but generation only)
#from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


# ----------------------------
# 1. Load API Key
# ----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")


# ----------------------------
# 2. Load Document
# ----------------------------
def load_document(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ----------------------------
# 3. Chunking Strategy
# ----------------------------
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)
    return chunks


# ----------------------------
# 4. Create Vector Store (FREE)
# ----------------------------
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore


# ----------------------------
# 5. Retrieve Top-K Documents
# ----------------------------
def retrieve_docs(vectorstore, query, k=2):
    docs = vectorstore.similarity_search(query, k=k)
    return docs


# ----------------------------
# 6. Generate Answer (OpenAI)
# ----------------------------
# def generate_answer(context, question):

#     llm = ChatOpenAI(
#         model="gpt-3.5-turbo",
#         temperature=0
#     )

#     prompt = f"""
# You are an AI assistant.
# Answer ONLY using the provided context.
# If answer not found, say "I don't know."

# Context:
# {context}

# Question:
# {question}

# Answer:
# """

#     response = llm.invoke(prompt)
#     return response.content

#generate ans using ollama
def generate_answer(context, question):

    llm = ChatOllama(
        #model="llama3", #heavy weight
        model="phi3", #lightweightmodel 
        temperature=0
    )

    prompt = f"""
You are an AI assistant.
Answer ONLY using the provided context.
If answer not found, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)
    return response.content


# ----------------------------
# 7. Main Flow
# ----------------------------
def main():

    print("Loading documents...")
    text = load_document("documents.txt")

    print("Splitting into chunks...")
    chunks = split_text(text)
    print(f"Total Chunks: {len(chunks)}")

    print("Creating vector store (local embeddings)...")
    vectorstore = create_vector_store(chunks)
    print("\n✅ RAG system ready! (Using Phi3 locally)")

    while True:

        question = input("\nAsk your question (type 'exit' to quit): ")

        if question.lower() == "exit":
            break

        print("Retrieving relevant documents...")
        docs = retrieve_docs(vectorstore, question)

        context = "\n".join([doc.page_content for doc in docs])

        print("Generating answer using phi3...")
        answer = generate_answer(context, question)

        print("\n📌 Final Answer:")
        print(answer)


if __name__ == "__main__":
    main()
    