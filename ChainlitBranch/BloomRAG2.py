#Customized Document Bloom RAG system


from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.llms import huggingface_hub
from langchain.embeddings import HuggingFaceEmbeddings
import chainlit as cl
from typing import List
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pinecone import Pinecone
from langchain.vectorstores import faiss
from langchain.vectorstores.chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.docstore.document import Document
from chainlit.types import AskFileResponse


load_dotenv()


MVP_Prompt = PromptTemplate(input_variables = ["question"], template ="""
    You are an educational design assistant specializing in Bloom's Taxonomy. Your task is to transform a set of input questions into questions aligned with each level of Bloom's Taxonomy (Remember, Understand, Apply, Analyze, Evaluate, and Create). For every input question, generate a question for each level of the taxonomy, ensuring the new questions remain relevant to the topic of the original question. Structure your output as follows:

Original Question: {question}
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
    """)
    

llm = huggingface_hub(
    repo_id='declare-lab/flan-alpaca-large', #declare-lab/flan-alpaca-gpt4-xl
    model_kwargs={'temperature': 0.2, 'max_length': 400})


index_name = "langchain-demo"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

namespaces = set()

welcome_message = """Welcome to the BloomAI QA demo! To get started: Upload a PDF of your problems/questions material
"""


def process_file(file: AskFileResponse):
    import tempfile

    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader

    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tempfile:
        if file.type == "text/plain":
            tempfile.write(file.content)
        elif file.type == "application/pdf":
            with open(tempfile.name, "wb") as f:
                f.write(file.content)

        loader = Loader(tempfile.name)
        documents = loader.load()
        docs = text_splitter.split_documents(documents)
        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"
        return docs


def get_docsearch(file: AskFileResponse):
    docs = process_file(file)

    # Save data in the user session
    cl.user_session.set("docs", docs)

    # Create a unique namespace for the file
    namespace = str(hash(file.content))
    docsearch = Chroma.from_documents(
        docs, embeddings
        )
    return docsearch

    # if namespace in namespaces:
    #     docsearch = Pinecone.from_existing_index(
    #         index_name=index_name, embedding=embeddings, namespace=namespace
    #     )
    # else:
    #     docsearch = Pinecone.from_documents(
    #         docs, embeddings, index_name=index_name, namespace=namespace
    #     )
    #     namespaces.add(namespace)

    # return docsearch


@cl.on_chat_start
async def start():
    await cl.Avatar(
        name="BloomAI",
        path = "./public/MoroccoAI.png" ,
    ).send()
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome_message,
            accept=["text/plain", "application/pdf"],
            max_size_mb=20,
            timeout=180,
            disable_human_feedback=True,
        ).send()

    file = files[0]

    msg = cl.Message(
        content=f"Processing `{file.name}`...", disable_human_feedback=True, author= "BloomAI"
    )
    await msg.send()
 
    # No async implementation in the Pinecone client, fallback to sync
    docsearch = await cl.make_async(get_docsearch)(file)

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key = "context",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )
    
    chain = ConversationalRetrievalChain.from_llm(llm=llm,chain_type = "stuff", retriever=docsearch.as_retriever(search_kwargs={'k': 2}),memory=memory,combine_docs_chain_kwargs={'prompt': MVP_Prompt},return_source_documents=True,verbose=True)

    # Let the user know that the system is ready
    msg.content = f"`{file.name}` processed. You can now ask questions! As your meta-tutor, I will try to help answering your questions as much as I can"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    # source_documents = res["source_documents"]  # type: List[Document]

    # text_elements = []  # type: List[cl.Text]

    # if source_documents:
    #     for source_idx, source_doc in enumerate(source_documents):
    #         source_name = f"source_{source_idx}"
    #         # Create the text element referenced in the message
    #         text_elements.append(
    #             cl.Text(content=source_doc.page_content, name=source_name)
    #         )
    #     source_names = [text_el.name for text_el in text_elements]

    #     if source_names:
    #         answer += f"\nSources: {', '.join(source_names)}"
    #     else:
    #         answer += "\nNo sources found"

    await cl.Message(content=answer, author ="BloomAI", disable_human_feedback=False).send()
    

    
    
