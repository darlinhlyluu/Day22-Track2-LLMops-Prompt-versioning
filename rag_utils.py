"""Reusable RAG helpers for the Day 22 lab scripts."""

from __future__ import annotations

from pathlib import Path

from config import DATA_DIR, Settings, make_chat_model, make_embeddings


def get_rag_prompts():
    from langchain_core.prompts import ChatPromptTemplate

    system_v1 = (
        "You are a faithful RAG assistant. Answer using ONLY facts that appear in the provided context. "
        "Write one or two short sentences. Do not add examples, caveats, dates, or explanations unless they are explicitly in the context. "
        "If the context is insufficient, say: I do not have enough information.\n\n"
        "Context:\n{context}"
    )
    system_v2 = (
        "You are an expert AI tutor, but faithfulness is the top priority. Use ONLY the provided context.\n"
        "Return exactly two compact lines:\n"
        "Answer: a direct answer using context wording as much as possible.\n"
        "Evidence: one short supporting fact copied or closely paraphrased from the context.\n"
        "Do not include outside knowledge, examples, extra assumptions, or hedging that is not in the context. "
        "If the context lacks the answer, write: I do not have enough information.\n\n"
        "Context:\n{context}"
    )
    return {
        "v1": ChatPromptTemplate.from_messages([("system", system_v1), ("human", "{question}")]),
        "v2": ChatPromptTemplate.from_messages([("system", system_v2), ("human", "{question}")]),
    }


def build_vectorstore(settings: Settings, source: Path | None = None):
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    source = source or DATA_DIR / "knowledge_base.txt"
    text = source.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    embeddings = make_embeddings(settings)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    print(f"Loaded {source} and indexed {len(chunks)} chunks with FAISS")
    return vectorstore


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def make_retriever(vectorstore, k: int = 3):
    return vectorstore.as_retriever(search_kwargs={"k": k})


def make_answer_chain(prompt, settings: Settings, temperature: float = 0.0):
    from langchain_core.output_parsers import StrOutputParser

    llm = make_chat_model(settings, temperature=temperature)
    return prompt | llm | StrOutputParser()
