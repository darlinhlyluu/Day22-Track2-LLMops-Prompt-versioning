# Day 22 Evidence Notes

Fill this folder while running the lab.

Required manual screenshots:

- `01_langsmith_traces.png`: LangSmith project run tab showing at least 50 Step 1 traces with inputs, retrieved context, answer, and latency visible.
- `02_prompt_hub.png`: Prompt Hub showing both prompt names, for example `day22-rag-prompt-v1` and `day22-rag-prompt-v2`.
- `03_ragas_scores.png`: Terminal comparison table from Step 3 showing V1 vs V2 and the `Target met` line.

Generated automatically by scripts:

- `02_ab_routing_log.txt`
- `03_ragas_report.json`
- `03_ragas_scores.txt`
- `04_pii_demo_log.txt`
- `04_json_demo_log.txt`

Brief analysis template:

- V1 is concise, so it may reduce unsupported extra claims.
- V2 is structured, so it may improve answer relevancy and readability.
- Submit the version with faithfulness >= 0.8 as evidence that the RAG pipeline is grounded.

