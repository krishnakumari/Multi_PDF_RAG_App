
import os
import tempfile
import hashlib
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader

from langchain_community.vectorstores import FAISS

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Multi PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

# --------------------------------------------------
# 2. LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error(
        "OPENAI_API_KEY not found. "
        "Please configure it in your .env file."
    )
    st.stop()


# --------------------------------------------------
# 3. INITIALIZE SESSION STATE
# --------------------------------------------------

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "file_signature" not in st.session_state:
    st.session_state.file_signature = None


# --------------------------------------------------
# 4. INITIALIZE MODELS
# --------------------------------------------------

@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )


@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=OPENAI_API_KEY
    )


embeddings = get_embeddings()
llm = get_llm()


# --------------------------------------------------
# 5. APPLICATION HEADER
# --------------------------------------------------

st.title("📚 Multi-PDF RAG Assistant")

st.markdown(
    """
    Upload multiple PDF documents and ask questions
    across all documents simultaneously.

    Every answer includes its source PDF and page number.
    """
)

st.divider()


# --------------------------------------------------
# 6. PDF UPLOAD
# --------------------------------------------------

st.subheader("📂 Upload PDF Documents")

uploaded_files = st.file_uploader(
    "Select multiple PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


# --------------------------------------------------
# 7. PROCESS PDF DOCUMENTS
# --------------------------------------------------

def process_pdfs(files):

    all_documents = []
    processed_names = []

    progress = st.progress(0)
    status = st.empty()

    total_files = len(files)

    for index, uploaded_file in enumerate(files):

        status.info(
            f"Processing: {uploaded_file.name}"
        )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        try:

            loader = PyPDFLoader(temp_path)

            documents = loader.load()

            for document in documents:

                # Preserve original PDF filename
                document.metadata["source"] = uploaded_file.name

                # PyPDFLoader uses zero-based page indexes.
                # Convert to human-readable page numbers.
                document.metadata["page_number"] = (
                    document.metadata.get("page", 0) + 1
                )

            all_documents.extend(documents)

            processed_names.append(uploaded_file.name)

        finally:

            if os.path.exists(temp_path):
                os.remove(temp_path)

        progress.progress(
            (index + 1) / total_files
        )

    status.info("Splitting documents into chunks...")

    # Split all documents together.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(
        all_documents
    )

    if not chunks:
        raise ValueError(
            "No readable text found in the uploaded PDFs."
        )

    status.info("Generating embeddings and building FAISS index...")

    # Create ONE FAISS vector database
    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    progress.progress(100)
    status.success("All PDFs processed successfully!")

    return vector_db, processed_names, len(chunks)


# --------------------------------------------------
# 8. PROCESS BUTTON
# --------------------------------------------------

if uploaded_files:

    # Detect changes in uploaded file collection
    signature_data = "|".join(
        f"{file.name}:{hashlib.md5(file.getvalue()).hexdigest()}"
        for file in sorted(uploaded_files, key=lambda x: x.name)
    )

    file_signature = hashlib.md5(
        signature_data.encode()
    ).hexdigest()

    if st.button(
        "🚀 Process All PDFs",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner("Building your knowledge base..."):

                vector_db, names, chunk_count = process_pdfs(
                    uploaded_files
                )

                st.session_state.vector_db = vector_db
                st.session_state.processed_files = names
                st.session_state.chat_history = []
                st.session_state.file_signature = file_signature

            st.success(
                f"Successfully processed {len(names)} PDFs "
                f"into {chunk_count} chunks!"
            )

        except Exception as error:

            st.error(f"Processing failed: {error}")


# --------------------------------------------------
# 9. DOCUMENT INFORMATION
# --------------------------------------------------

if st.session_state.vector_db:

    st.divider()

    st.subheader("📊 Knowledge Base Information")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Total PDF Documents",
            len(st.session_state.processed_files)
        )

    with col2:

        st.metric(
            "Vector Database",
            "FAISS"
        )

    with st.expander("View Processed Documents"):

        for filename in st.session_state.processed_files:
            st.write(f"📄 {filename}")


# --------------------------------------------------
# 10. RAG RETRIEVAL FUNCTION
# --------------------------------------------------

def retrieve_documents(question, k=5):

    results = st.session_state.vector_db.similarity_search(
        question,
        k=k
    )

    return results


# --------------------------------------------------
# 11. GENERATE ANSWER
# --------------------------------------------------

def generate_answer(question):

    # Retrieve relevant chunks across all PDFs
    retrieved_docs = retrieve_documents(
        question,
        k=5
    )

    if not retrieved_docs:
        return (
            "I could not find relevant information "
            "in the uploaded documents."
        ), []

    # Prepare context with source metadata
    context_parts = []

    for index, doc in enumerate(retrieved_docs, start=1):

        filename = doc.metadata.get(
            "source",
            "Unknown file"
        )

        page = doc.metadata.get(
            "page_number",
            "Unknown"
        )

        context_parts.append(
            f"""
            [Source {index}]
            File: {filename}
            Page: {page}

            Content:
            {doc.page_content}
            """
        )

    context = "\n\n".join(context_parts)

    # Prompt template
    prompt = ChatPromptTemplate.from_template(
        """
        You are an intelligent document assistant.

        Answer the user's question using ONLY the
        information provided in the context.

        RULES:

        1. Do not invent information.
        2. If the answer is not available in the
           context, clearly say so.
        3. Cite every factual claim using the exact
           source format:
           [filename, Page X]
        4. If information comes from multiple PDFs,
           cite all relevant sources.
        5. Do not create fake page numbers.
        6. Keep the answer clear and structured.

        CONTEXT:
        {context}

        USER QUESTION:
        {question}

        ANSWER:
        """
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    return response.content, retrieved_docs


# --------------------------------------------------
# 12. CHAT INTERFACE
# --------------------------------------------------

st.divider()

st.subheader("💬 Ask Questions Across Your PDFs")

if not st.session_state.vector_db:

    st.info(
        "Please upload and process your PDF documents "
        "before asking questions."
    )

else:

    # Display chat history
    for chat in st.session_state.chat_history:

        with st.chat_message("user"):
            st.write(chat["question"])

        with st.chat_message("assistant"):

            st.markdown(chat["answer"])

            with st.expander("📚 View Source Documents"):

                for source in chat["sources"]:

                    filename = source.metadata.get(
                        "source",
                        "Unknown"
                    )

                    page = source.metadata.get(
                        "page_number",
                        "Unknown"
                    )

                    st.markdown(
                        f"**📄 File:** {filename}"
                    )

                    st.markdown(
                        f"**📃 Page:** {page}"
                    )

                    st.caption(
                        source.page_content
                    )

                    st.divider()

    # User question input
    question = st.chat_input(
        "Ask anything about your uploaded PDFs..."
    )

    if question:

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):

            with st.spinner("Searching documents and generating answer..."):

                try:

                    answer, sources = generate_answer(
                        question
                    )

                    st.markdown(answer)

                    with st.expander("📚 View Source Documents"):

                        for source in sources:

                            filename = source.metadata.get(
                                "source",
                                "Unknown"
                            )

                            page = source.metadata.get(
                                "page_number",
                                "Unknown"
                            )

                            st.markdown(
                                f"**📄 File:** {filename}"
                            )

                            st.markdown(
                                f"**📃 Page:** {page}"
                            )

                            st.caption(
                                source.page_content
                            )

                            st.divider()

                    st.session_state.chat_history.append(
                        {
                            "question": question,
                            "answer": answer,
                            "sources": sources
                        }
                    )

                except Exception as error:

                    st.error(
                        f"Error generating answer: {error}"
                    )


# --------------------------------------------------
# 13. SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("⚙️ Settings")

    st.markdown("### RAG Configuration")

    st.write("**Embedding Model**")
    st.code("text-embedding-3-small")

    st.write("**LLM Model**")
    st.code("gpt-4o-mini")

    st.write("**Vector Database**")
    st.code("FAISS")

    st.write("**Chunk Size**")
    st.write("1000 characters")

    st.write("**Chunk Overlap**")
    st.write("200 characters")

    st.write("**Retrieved Documents**")
    st.write("Top 5")

    st.divider()

    if st.button(
        "🗑️ Clear Knowledge Base",
        use_container_width=True
    ):

        st.session_state.vector_db = None
        st.session_state.processed_files = []
        st.session_state.chat_history = []
        st.session_state.file_signature = None

        st.rerun()

    st.caption(
        "Multi-PDF RAG Assistant | LangChain + FAISS"
    )
