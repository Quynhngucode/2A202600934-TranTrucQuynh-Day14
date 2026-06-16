# Individual Reflection - Tran Truc Quynh

## Dong gop ky thuat

- Hoan thien synthetic golden dataset 60 cases co ground-truth document IDs va chunk IDs.
- Xay dung agent RAG offline co retrieval, citation, token usage va estimated cost.
- Implement retrieval metrics: Hit Rate, MRR, Recall va Precision.
- Implement async benchmark runner cho toan bo golden dataset.
- Implement multi-judge consensus voi hai judge doc lap, agreement rate va conflict handling.
- Them regression gate so sanh `v1_base` voi `v2_hybrid_rerank`.
- Tao reports bat buoc: `reports/summary.json`, `reports/benchmark_results.json`, va `analysis/failure_analysis.md`.

## Bai hoc rut ra

Retrieval metrics phai duoc do rieng truoc generation vi hallucination thuong bat dau tu viec lay sai context. MRR giup nhin ro ground-truth chunk co nam o vi tri dau hay bi day xuong duoi. Multi-judge consensus lam ket qua cham diem on dinh hon so voi viec tin mot judge don le.

## Huong cai tien

- Thay offline judge bang hai model that khi co API key.
- Them synonym dictionary tieng Viet co dau/khong dau cho retriever.
- Thu nghiem cross-encoder reranker de cai thien hard cases.
- Bo sung cost dashboard neu benchmark duoc chay thuong xuyen.
