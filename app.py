import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import tempfile
import torch
import os

st.set_page_config(page_title="RAG Chatbot PDF", layout="wide")
st.title("📄 RAG Chatbot from PDF (Chat Style + Memory)")

# Session state for storing chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Upload PDF
uploaded_file = st.file_uploader("📤 Upload a PDF file", type=["pdf"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.success("✅ PDF uploaded and processing...")

    # Load and chunk PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Embedding + Vector DB
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embedding_model)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Load LLM model 
    model_name = "scb10x/typhoon2.1-gemma3-4b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="auto")

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=False,
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # Prompt Template
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Answer the question based **only** on the information provided below.\n"
            "Do not repeat or quote the context. Keep your answer short and relevant.\n"
            "If the answer is not in the context, just say 'I don't know.'\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        )
    )

    # Memory + Chain
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=custom_prompt,
        return_source_documents=False,
    )

    st.markdown("---")
    st.markdown("### 💬 Chat with your PDF")

    # chat history
    for chat in st.session_state.chat_history:
        role = chat["role"]
        message = chat["content"]
        with st.chat_message(role):
            st.markdown(message)

    # Ask new question
    user_input = st.chat_input("Ask something...")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("🤖 Thinking..."):
            result = qa_chain({"question": user_input})
            answer = result["answer"]

        with st.chat_message("assistant"):
            st.markdown(answer)

        # Save to session_state as dict for consistency with st.chat_message role
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # os.remove(pdf_path)
