from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import chromadb
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
folder_path = "./data/"
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(api_key=api_key)

# Bước 1: Load dữ liệu từ file văn bản
def initialize_chroma():
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="document_store")
    return collection

def load_and_split_documents(folder_path):
    text_loader_kwargs={'autodetect_encoding': True}
    loader = DirectoryLoader(folder_path, glob="./*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    
    documents = loader.load()  # Trả về danh sách tài liệu
    
    if not documents:
        print("Không tìm thấy tài liệu nào trong thư mục!")
        return []

    # Tách nhỏ tài liệu để phù hợp với mô hình NLP
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    split_docs = text_splitter.split_documents(documents)
    return split_docs

def add_documents_to_chroma(collection, docs, embeddings):
    for i, doc in enumerate(docs):
        # Tạo embedding vector
        vector = embeddings.embed_query(doc.page_content)
        embedding_id = str(i)  # ID embedding là chuỗi của chỉ số
        
        # Thêm hoặc cập nhật tài liệu (upsert)
        collection.upsert(
            ids=[embedding_id],
            embeddings=[vector],
            metadatas=[{"source": doc.metadata["source"]}],
            documents=[doc.page_content]
        )

def query_chroma(collection, query, embeddings):
    query_vector = embeddings.embed_query(query)
    results = collection.query(query_embeddings=[query_vector], n_results=10)
    return results

# Tạo PromptTemplate cho câu hỏi
prompt_template = """
Bạn là một trợ lý tuyệt vời
Dựa trên các tài liệu dưới đây, trả lời câu hỏi một cách thân thiện và chi tiết:
Nếu không có tài liệu thì cứ chò chuyện trao đổi bình thường
Tài liệu:
{documents}


Câu hỏi: {question}
Trả lời:
"""

def chatbot(folder_path):
    collection = initialize_chroma()
    if collection.count() == 0:
        print("Đang tải và xử lý tài liệu...")
        docs = load_and_split_documents(folder_path)
        add_documents_to_chroma(collection, docs, embeddings)

    print("Chatbot đã sẵn sàng. Nhập câu hỏi của bạn:")
    # Tạo PromptTemplate và nối với LLM trực tiếp bằng pipe
    prompt = PromptTemplate(input_variables=["documents", "question"], template=prompt_template)
    # Thực hiện pipe giữa PromptTemplate và LLM
    llm_pipe = prompt | llm  

    while True:
        query = input("Bạn: ")
        if query.lower() in ["exit", "quit", "thoát"]:
            print("Chatbot: Tạm biệt!")
            break
        results = query_chroma(collection, query, embeddings)
        print(results["distances"][0])
        print(min(results["distances"][0]))
        
        if min(results["distances"][0]) < 0.36:
            documents_content = "\n".join([str(doc) for doc in results["documents"]])  # Đảm bảo mỗi phần tử là chuỗi
            response = llm_pipe.invoke({"documents": documents_content, "question": query})
            print("Chatbot:", response.content)
        else:
            response = llm_pipe.invoke({"documents": [], "question": query})
            print("Chatbot:", response.content)

if __name__ == "__main__": 
    chatbot(folder_path)
 
