import streamlit as st
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import pytesseract
from PIL import Image
import io

st.set_page_config(page_title="YONO AI Assistant", layout="wide")
st.title("📘 YONO AI – User Story Assistant")

# Extract content from DOCX
def extract_docx(file):
    doc = Document(file)
    content = []

    for para in doc.paragraphs:
        if para.text.strip():
            content.append(para.text)

    for table in doc.tables:
        for row in table.rows:
            row_text = [cell.text.strip() for cell in row.cells]
            content.append(" | ".join(row_text))

    for rel in doc.part.rels:
        if "image" in doc.part.rels[rel].target_ref:
            image_data = doc.part.rels[rel]._target.blob
            image = Image.open(io.BytesIO(image_data))
            text = pytesseract.image_to_string(image)
            content.append(text)

    return "\n".join(content)

# Split text
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# Create vector DB
def create_vector_db(chunks):
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_texts(chunks, embedding=embeddings, persist_directory="./db")
    return vectordb

# Create QA chain
def create_qa(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0),
        retriever=retriever,
        return_source_documents=True
    )
    return qa

uploaded_file = st.file_uploader("Upload User Story DOCX", type=["docx"])

if uploaded_file:
    with st.spinner("Processing document..."):
        text = extract_docx(uploaded_file)
        chunks = split_text(text)
        vectordb = create_vector_db(chunks)
        qa = create_qa(vectordb)

    st.success("Document processed successfully!")

    query = st.text_input("Ask your question about user stories")

    if query:
        result = qa({"query": query})
        st.subheader("Answer")
        st.write(result["result"])

        st.subheader("Sources")
        for doc in result["source_documents"]:
            st.write(doc.page_content[:300])
