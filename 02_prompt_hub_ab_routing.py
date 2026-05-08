"""Step 2: LangSmith Prompt Hub plus deterministic A/B routing."""

from __future__ import annotations

import hashlib
from collections import Counter

from config import EVIDENCE_DIR, configure_langsmith, ensure_dirs, require_api_settings

settings = configure_langsmith()

from langsmith import Client, traceable

from qa_pairs import SAMPLE_QUESTIONS
from rag_utils import build_vectorstore, get_rag_prompts, make_answer_chain, make_retriever

PROMPTS = get_rag_prompts()
PROMPT_V1_NAME = settings.prompt_v1_name
PROMPT_V2_NAME = settings.prompt_v2_name


def push_prompts_to_hub(client: Client) -> None:
    for name, prompt, description in [
        (PROMPT_V1_NAME, PROMPTS["v1"], "Day 22 RAG prompt V1: concise answers"),
        (PROMPT_V2_NAME, PROMPTS["v2"], "Day 22 RAG prompt V2: structured tutor answers"),
    ]:
        try:
            url = client.push_prompt(name, object=prompt, description=description)
            print(f"Pushed {name}: {url}")
        except Exception as exc:
            print(f"Prompt push skipped or updated for {name}: {exc}")


def pull_prompts_from_hub(client: Client) -> dict[str, object]:
    pulled = {}
    for key, name in [("v1", PROMPT_V1_NAME), ("v2", PROMPT_V2_NAME)]:
        try:
            pulled[key] = client.pull_prompt(name)
            print(f"Pulled {name} from Prompt Hub")
        except Exception as exc:
            pulled[key] = PROMPTS[key]
            print(f"Using local fallback for {name}: {exc}")
    return pulled


def get_prompt_version(request_id: str) -> str:
    hash_int = int(hashlib.md5(request_id.encode("utf-8")).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, answer_chains: dict[str, object], question: str, version_name: str) -> dict:
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    version_key = "v1" if version_name == PROMPT_V1_NAME else "v2"
    answer = answer_chains[version_key].invoke(
        {"context": "\n\n".join(contexts), "question": question}
    )
    return {
        "question": question,
        "version": version_key,
        "prompt_name": version_name,
        "retrieved_contexts": contexts,
        "answer": answer,
    }


def main() -> None:
    ensure_dirs()
    require_api_settings(settings)

    print("=" * 60)
    print("  Step 2: Prompt Hub A/B Routing")
    print("=" * 60)

    client = Client(api_key=settings.langsmith_api_key)
    push_prompts_to_hub(client)
    prompts = pull_prompts_from_hub(client)

    vectorstore = build_vectorstore(settings)
    retriever = make_retriever(vectorstore, k=3)
    answer_chains = {
        "v1": make_answer_chain(prompts["v1"], settings),
        "v2": make_answer_chain(prompts["v2"], settings),
    }

    lines = []
    counts = Counter()
    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        request_id = f"req-{i:04d}"
        version_name = get_prompt_version(request_id)
        result = ask_ab(retriever, answer_chains, question, version_name)
        counts[result["version"]] += 1
        line = f"[{i:02d}] [prompt-{result['version']}] {request_id} -> {question}"
        lines.append(line)
        print(line)
        print(f"     {result['answer'][:160]}")

    summary = f"Routing summary: prompt-v1={counts['v1']}, prompt-v2={counts['v2']}"
    lines.append(summary)
    print(summary)

    log_path = EVIDENCE_DIR / "02_ab_routing_log.txt"
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {log_path}")
    print("Capture Prompt Hub as evidence/02_prompt_hub.png.")


if __name__ == "__main__":
    main()

