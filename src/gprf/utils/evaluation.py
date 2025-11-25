from tqdm import tqdm

def construct_final_query(original_query, cets_gen, prf_terms):
    """合并原始查询和扩展词"""
    combined_terms = list(set(cets_gen + prf_terms))  # 合并并去重
    final_query = original_query + " " + " ".join(combined_terms)
    return final_query

# ----------------评估结果-----------------------------
def evaluate_retrieval_performance(test_dataset, dpr_retriever, rm3_retriever, bart_generator, prf_expander, config):
    """评估检索性能"""
    # DPR评估结果
    dpr_correct_count_topk = 0
    dpr_em_correct_count = 0

    # RM3评估结果
    rm3_correct_count_topk = 0
    rm3_em_correct_count = 0

    total_count = len(test_dataset)
    top_k = config.get('retrieval', {}).get('top_k', 100)
    expansion_terms = config.get('retrieval', {}).get('expansion_terms', 5)

    for example in tqdm(test_dataset, desc="Evaluating"):
        original_query = example["Question"]

        # 生成扩展查询
        cets_gen = bart_generator.generate_expansion_batch([example], max_length=expansion_terms)[0].split()
        prf_terms = prf_expander.get_prf_terms(original_query)
        final_query = construct_final_query(original_query, cets_gen, prf_terms)

        # DPR检索和评估
        dpr_hits = dpr_retriever.retrieve([final_query], top_k=top_k)
        dpr_docids = [hit.docid for hit in dpr_hits[0]]
        gold_answer = example.get('GoldDocID', None)

        if gold_answer:
            if gold_answer in dpr_docids:
                dpr_correct_count_topk += 1
            if gold_answer == dpr_docids[0]:
                dpr_em_correct_count += 1

        # RM3检索和评估
        rm3_hits = rm3_retriever.retrieve([final_query], top_k=top_k)
        rm3_docids = [hit.docid for hit in rm3_hits[0]]

        if gold_answer:
            if gold_answer in rm3_docids:
                rm3_correct_count_topk += 1
            if gold_answer == rm3_docids[0]:
                rm3_em_correct_count += 1

    # 计算准确率
    dpr_accuracy_topk = dpr_correct_count_topk / total_count * 100
    dpr_em_accuracy = dpr_em_correct_count / total_count * 100
    rm3_accuracy_topk = rm3_correct_count_topk / total_count * 100
    rm3_em_accuracy = rm3_em_correct_count / total_count * 100

    print(f"DPR Top-{top_k} Accuracy: {dpr_accuracy_topk:.2f}%")
    print(f"DPR EM Accuracy: {dpr_em_accuracy:.2f}%")
    print(f"RM3 Top-{top_k} Accuracy: {rm3_accuracy_topk:.2f}%")
    print(f"RM3 EM Accuracy: {rm3_em_accuracy:.2f}%")

    return dpr_accuracy_topk, dpr_em_accuracy, rm3_accuracy_topk, rm3_em_accuracy
