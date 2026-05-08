"""Step 1: LangSmith-instrumented RAG pipeline."""

from __future__ import annotations

from config import configure_langsmith, ensure_dirs, require_api_settings

settings = configure_langsmith()

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langsmith import traceable

from qa_pairs import SAMPLE_QUESTIONS
from rag_utils import build_vectorstore as _build_vectorstore
from rag_utils import format_docs, get_rag_prompts, make_retriever
from config import make_chat_model


def build_vectorstore():
    return _build_vectorstore(settings)


def build_rag_chain(vectorstore):
    retriever = make_retriever(vectorstore, k=3)
    prompt = get_rag_prompts()["v1"]
    llm = make_chat_model(settings, temperature=0.0)
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | RunnablePassthrough.assign(answer=prompt | llm | StrOutputParser())
    )
    return chain, retriever


@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> dict:
    result = chain.invoke(question)
    return {
        "question": result["question"],
        "retrieved_context": result["context"],
        "answer": result["answer"],
    }


def main() -> None:
    ensure_dirs()
    require_api_settings(settings)

    print("=" * 60)
    print("  Step 1: LangSmith RAG Pipeline")
    print("=" * 60)
    print(f"LangSmith project: {settings.langsmith_project}")

    vectorstore = build_vectorstore()
    chain, _ = build_rag_chain(vectorstore)

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        result = ask(chain, question)
        print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] Q: {question}")
        print(f"       A: {result['answer'][:180]}\n")

    print(f"Step 1 complete: sent {len(SAMPLE_QUESTIONS)} traced RAG calls to LangSmith.")
    print("Open https://smith.langchain.com and capture evidence/01_langsmith_traces.png.")


if __name__ == "__main__":
    main()

