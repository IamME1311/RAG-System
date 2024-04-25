import streamlit as st


from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Data/leave-no-context-behind.pdf")

data = loader.load_and_split()


from langchain_text_splitters import NLTKTextSplitter

text_splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)

chunks = text_splitter.split_documents(data)


f = open("Data/api_key.txt")

key = f.read()


from langchain_google_genai import GoogleGenerativeAIEmbeddings

embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=key, 
                                               model="models/embedding-001")


from langchain_community.vectorstores import Chroma

# Embed each chunk and load it into the vector store
db = Chroma.from_documents(chunks, embedding_model, persist_directory="./RAG_DB")

# Persist the database on drive
db.persist()

db_connection = Chroma(persist_directory="./RAG_DB", embedding_function=embedding_model)

retriever = db_connection.as_retriever(search_kwargs={"k": 5})


from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

chat_template = ChatPromptTemplate.from_messages([
    # System Message Prompt Template
    SystemMessage(content="""You are a Helpful AI Bot. 
    You take the context and question from user. Your answer should be based on the specific context."""),
    # Human Message Prompt Template
    HumanMessagePromptTemplate.from_template("""Aswer the question based on the given context.
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer: """)
])


from langchain_google_genai import ChatGoogleGenerativeAI

chat_model = ChatGoogleGenerativeAI(google_api_key=key, 
                                   model="gemini-1.5-pro-latest")


from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()

from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)


st.header("My first RAG")


user_input = st.text_input("Enter what you want to know..")
if st.button("Generate")== True:
    response = rag_chain.invoke(user_input)
    st.markdown(response)