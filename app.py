import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import tempfile
import os
import torch
from langchain.prompts import PromptTemplate  
st.title("📄 RAG QA from PDF with Streamlit")


uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
if uploaded_file is not None:
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.success("PDF uploaded! Processing...")

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embedding_model)

    model_name = "scb10x/typhoon2.1-gemma3-4b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="auto")


    custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful assistant. Use the information below to answer the user's question.\n"
        "If you don't know the answer, just say 'I don't know.'\n\n"
        "{context}\n\n"
        "{question}"
    )
)


    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=1.0,
        do_sample=False,
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",chain_type_kwargs={"prompt": custom_prompt},retriever=retriever)

    query = st.text_input("Ask a question about your PDF:")
    if query:
        with st.spinner("Thinking..."):
            answer = qa_chain.run(query)
        st.markdown(f"### Answer:\n{answer}")

    # Clean up temp file after use (optional)
    # os.remove(pdf_path)
