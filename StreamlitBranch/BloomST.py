import os
import sys
import keys
import tokenization 
import config
import openai 
import faiss


from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, Pinecone, faiss, FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.llms import huggingface_hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.docstore.document import Document


os.environ["OPENAI_API_KEY"] = keys.key
llm = ChatOpenAI(model_name=config.MODEL_NAME, temperature=0, openai_api_key=keys.key)



import streamlit as st
from typing import List
import tempfile

from dotenv import load_dotenv

load_dotenv()

# from langchain_community.llms import OpenLLM
# from langchain_core.prompts import PromptTemplate
# from langchain_mistralai import ChatMistralAI
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.llms import huggingface_hub
# from langchain_huggingface.llms import HuggingFaceLLM
# from langchain_community.document_loaders import PyPDFLoader, TextLoader
# from langchain_community.vectorstores import Pinecone
# from langchain_community.vectorstores import Chroma
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_community.llms.gpt4all import GPT4All 
# from langchain_community.models.huggingface
# from langchain_huggingface.llms

# from langchain_mistralai import ChatMistralAI

# llm = ChatMistralAI(
#     model="mistral-large-latest",
#     temperature=0,
#     max_retries=2,
#     # other params...
# )






# Prompt for Bloom's Taxonomy questions
MVP_Prompt = PromptTemplate(
    input_variables=["context"],
    template="""
    You are an educational design assistant specializing in Bloom's Taxonomy. Your task is to transform a set of input questions into questions aligned with each level of Bloom's Taxonomy (Remember, Understand, Apply, Analyze, Evaluate, and Create). For every input question, generate a question for each level of the taxonomy, ensuring the new questions remain relevant to the topic of the original question. Structure your output as follows:

Original Question: {context}
Remember: [Question targeting recall of knowledge]
Understand: [Question requiring comprehension]
Apply: [Question involving practical application]
Analyze: [Question prompting breakdown into components]
Evaluate: [Question asking for judgment or critique]
Create: [Question requiring synthesis or creation of new ideas]
Example:
Input Question: What are the causes of climate change?

Remember: What is climate change, and what are its primary causes?
Understand: How do greenhouse gases contribute to climate change?
Apply: Can you identify the greenhouse gas emissions in your daily activities?
Analyze: What are the key differences between natural and human-induced causes of climate change?
Evaluate: How effective are current policies in mitigating climate change?
Create: Propose a new strategy to reduce the effects of climate change on urban areas
    """
)

# Initialize LLM and embeddings
# llm = HuggingFaceLLM(
#     repo_id='declare-lab/flan-alpaca-large',
#     model_kwargs={'temperature': 0.2, 'max_length': 400}
# )

# embeddings = HuggingFaceEmbeddings(
#     model_name='sentence-transformers/all-MiniLM-L6-v2',
#     model_kwargs={'device': 'cpu'}
# )

embeddings = OpenAIEmbeddings(model=config.MODEL_NAME, openai_api_key=keys.key)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# File processing function
def process_file(file) -> List:
    file_type = file.type
    file_content = file.read()

    # Determine loader based on file type
    Loader = TextLoader if file_type == "text/plain" else PyPDFLoader

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file_content)
        temp_file.close()

        loader = Loader(temp_file.name)
        documents = loader.load()
        docs = text_splitter.split_documents(documents)
        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"
        return docs

# Build document search      docsearch = FAISS.from_texts(texts,embeddings)     # docsearch = Chroma.from_documents(documents=docs, embedding= embeddings)
# def get_docsearch(file) -> Chroma:
#     docs = process_file(file)
#     texts = [doc.page_content for doc in docs]
#     docsearch = FAISS.from_documents(texts,embeddings)
#     return docsearch


def get_docsearch(file) -> FAISS:
    docs = process_file(file)
    documents = [Document(page_content=doc.page_content, metadata=doc.metadata) for doc in docs]
    docsearch = FAISS.from_documents(documents, embeddings)
    return docsearch


# Streamlit app
st.image("./public/MoroccoAI.png")
st.title("BloomAI QA Demo")

# Sidebar for file upload
st.sidebar.header("Upload Your File")
uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF or text file containing your questions.",
    type=["pdf", "txt"]
)

if uploaded_file:
    st.sidebar.write("Processing your file...")
    docsearch = get_docsearch(uploaded_file)
    st.sidebar.success("File processed successfully!")

    # Create memory and chat chain
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="context",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
        memory=memory,
        combine_docs_chain_kwargs={'prompt': MVP_Prompt},
        return_source_documents=True,
        verbose=True
    )

    # User input and chat interface
    st.header("Answer the questions")
    user_input = st.text_input("Answer all your questions in a list format:", "")

    if user_input:
        with st.spinner("Generating response..."):
            response = chain.run(user_input)
            st.success("Response Generated!")
            st.write(response)

    # Display sources (optional)
    st.subheader("Sources")
    for doc in docsearch.similarity_search(user_input, k=2):
        st.write(f"Source: {doc.metadata['source']}")
        st.write(doc.page_content)
else:
    st.info("Please upload your practice problems to begin.")
