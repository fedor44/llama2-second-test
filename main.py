from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
#from numpy.linalg import norm
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.llms import CTransformers
from langchain_community.embeddings import JinaEmbeddings
from langchain.chains import RetrievalQA

# set huggingface access token
#os.environ['HF_TOKEN'] = "hf_*"

# load pdf file for chat
loader = PyPDFLoader("./llama2.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=500,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.split_documents(pages)

# cosine similarity calculator 
#cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))

# set embedding model(jina)
embeddings = JinaEmbeddings(jina_api_key='jina_*', model_name='jina-embeddings-v2-base-en')

# save vector data to vectorstore(Chroma)
vectorstore = Chroma.from_documents(texts, embeddings)

question = "What is the llama2?"
# load llama2 model
llm = CTransformers(
    model="llama-2-7b-chat.ggmlv3.q4_K_S.bin",
    model_type="llama"
)
#retriever_from_llm = MultiQueryRetriever.from_llm(
#    retriever = vectorstore.as_retriever(), llm=llm
#)
#docs = retriever_from_llm.get_relevant_documents(query=question)

qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever()) 
result = qa_chain({"query": question})
print(result)