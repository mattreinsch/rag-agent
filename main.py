import os
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader

# Set your OpenAI API key
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# 1. Load documents
loader = DirectoryLoader('documents', glob="**/*.txt")
documents = loader.load()

# 2. Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# 3. Create a FAISS vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(texts, embeddings)

# 4. Set up the RAG chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# 5. Interactive loop to ask questions
while True:
    query = input("Ask a question: ")
    if query.lower() == 'exit':
        break
    print(qa.run(query))
