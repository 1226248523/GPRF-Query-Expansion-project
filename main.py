import logging
import json
import os
import time
import torch
from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration, DPRQuestionEncoderTokenizer, DPRQuestionEncoder
from pyserini.search.faiss import FaissSearcher
from pyserini.search.lucene import LuceneSearcher
from pyserini.analysis import Analyzer, get_lucene_analyzer
from datasets import load_from_disk

# 设置环境变量
os.environ['TORCH_HOME'] = '/mnt/data/torch-models'
os.environ['HF_HOME'] = '/mnt/data/huggingface'
os.environ["JAVA_TOOL_OPTIONS"] = "-Xms512m -Xmx8g" 

logging.basicConfig(
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)

print("PyTorch version:", torch.__version__)
print("Is CUDA available?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version used by PyTorch:", torch.version.cuda)
    print("Number of GPUs:", torch.cuda.device_count())
    print("Current GPU name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")
    
# ---------------BART Query Generator---------------
class BartQueryGenerator:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.tokenizer = BartTokenizer.from_pretrained(config['output_dir'])
        self.model = BartForConditionalGeneration.from_pretrained(config["output_dir"]).to(self.device)
        self.batch_size = config["bart_batch_size"]

    def format_input(self, example):
        answer = example.get('Answer', '[No Answer]')
        title = example.get('Title', '[No Title]')
        sentence = example.get('Sentence', '[No Sentence]')
        question = example.get('Question', '[No Question]')
        return f"{question} Answer: {answer} Title: {title} Sentence: {sentence}"

    def generate_expansion_batch(self, examples, max_length=10):
        inputs = [self.format_input(ex) for ex in examples]

        encoding = self.tokenizer(
            inputs,
            max_length=config["bart_max_length"],
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                encoding.input_ids,
                max_length=max_length,
                num_beams=1,
                early_stopping=True
            )

        expansions = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return expansions

# ---------------DPR Retriever-----------------
class DPRRetriever:
    def __init__(self, config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # 初始化 DPR 查询编码器
        self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(config["dpr_question_encoder"])
        self.model = DPRQuestionEncoder.from_pretrained(config["dpr_question_encoder"]).to(self.device)

        # 加载 Faiss 索引目录
        self.index_dir = config["dpr_index"]  
        self.searcher = FaissSearcher(self.index_dir)

        # 配置参数
        self.max_length = config["dpr_max_length"]
        self.batch_size = config["dpr_batch_size"]

    def encode_query(self, query):
        """ 将查询编码为 dense vector """
        inputs = self.tokenizer(
            [query],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            embeddings = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            ).pooler_output

        return embeddings.cpu().numpy()

    def retrieve(self, queries, top_k=100):
        """ 使用 DPR 编码 + Dense Searcher 进行检索 """
        results = []
        for query in queries:
            dense_vector = self.encode_query(query)
            hits = self.searcher.search(dense_vector, k=top_k)
            results.append(hits)
        return results

# ---------------RM3 Retriever-----------------
class RM3Retriever:
    def __init__(self, config):
        self.index_path = config["rm3_index"]
        self.searcher = LuceneSearcher(self.index_path)

    def retrieve(self, queries, top_k=100):
        results = []
        for query in queries:
            hits = self.searcher.search(query, k=top_k)
            results.append(hits)
        return results

# ---------------PRF Expander------------------
class PRFExpander:
    def __init__(self, index_path):
        # 使用 LuceneSearcher
        self.searcher = LuceneSearcher(index_path)
        self.searcher.set_bm25(k1=0.9, b=0.4)  # 设置 BM25 参数

        # 启用 RM3 查询扩展模型
        self.searcher.set_rm3(
            fb_terms=config['expansion_terms'],  # 扩展词数量
            fb_docs=10,                          # 使用前 10 篇文档作为初检结果
            original_query_weight=0.5           # 原始查询权重
        )

        self.analyzer = Analyzer(get_lucene_analyzer(language='en', stemming=True, stemmer='porter', stopwords=True))
        self._doc_cache = {}
        self.max_retries = 1
        self.retry_delay = 1

    def preprocess_query(self, query):
        return ''.join([i if ord(i) < 128 else ' ' for i in query])

    def safe_search(self, query, k):
        query = self.preprocess_query(query).strip()
        if not query:
            return []

        MAX_QUERY_LENGTH = 100
        query = " ".join(query.split()[:MAX_QUERY_LENGTH])  # 控制输入长度

        for attempt in range(self.max_retries):
            try:
                hits = self.searcher.search(query, k)
                return hits
            except Exception as e:
                logger.warning(f"[PRF] 检索失败 (query: '{query}') - 尝试 {attempt + 1}/{self.max_retries}, 错误: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return []
        return []

    def get_prf_terms(self, query):
        """ 使用 RM3 模型获取扩展词 """
        hits = self.safe_search(query, config['prf_top_n'])  # 获取前10篇文档

        term_counts = {}
        for hit in hits:
            try:
                doc_id = hit.docid
                if doc_id in self._doc_cache:
                    content = self._doc_cache[doc_id]
                else:
                    doc = self.searcher.doc(doc_id)
                    raw_content = doc.raw()
                    doc_json = json.loads(raw_content)
                    content = doc_json.get("contents", "")
                    self._doc_cache[doc_id] = content

                tokens = self.analyzer.analyze(content)
                for token in tokens:
                    term_counts[token] = term_counts.get(token, 0) + 1
            except Exception as e:
                logger.warning(f"解析文档时出错: {e}")

        query_terms = set(self.analyzer.analyze(query))
        expanded_terms = sorted(
            [k for k in term_counts.keys() if k not in query_terms],
            key=lambda x: term_counts[x],
            reverse=True
        )[:config['expansion_terms']]

        return expanded_terms

# ---------------合并CETS-GEN，构造最终查询Q'-----------------
def construct_final_query(original_query, cets_gen, prf_terms):
    combined_terms = list(set(cets_gen + prf_terms))  # 合并并去重
    final_query = original_query + " " + " ".join(combined_terms)
    return final_query

# ----------------评估结果-----------------------------
def evaluate_retrieval_performance(test_dataset, dpr_retriever, rm3_retriever, bart_generator, prf_expander, config):
    # DPR评估结果
    dpr_correct_count_topk = 0
    dpr_em_correct_count = 0
    
    # RM3评估结果  
    rm3_correct_count_topk = 0
    rm3_em_correct_count = 0
    
    total_count = len(test_dataset)

    for example in tqdm(test_dataset, desc="Evaluating"):
        original_query = example["Question"]
        
        # 生成扩展查询
        cets_gen = bart_generator.generate_expansion_batch([example], max_length=config["expansion_terms"])[0].split()
        prf_terms = prf_expander.get_prf_terms(original_query)
        final_query = construct_final_query(original_query, cets_gen, prf_terms)
        
        # DPR检索和评估
        dpr_hits = dpr_retriever.retrieve([final_query], top_k=config["top_k"])
        dpr_docids = [hit.docid for hit in dpr_hits[0]]
        gold_answer = example.get('GoldDocID', None)
        
        if gold_answer:
            if gold_answer in dpr_docids:
                dpr_correct_count_topk += 1
            if gold_answer == dpr_docids[0]:
                dpr_em_correct_count += 1
        
        # RM3检索和评估
        rm3_hits = rm3_retriever.retrieve([final_query], top_k=config["top_k"])
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
    
    print(f"DPR Top-{config['top_k']} Accuracy: {dpr_accuracy_topk:.2f}%")
    print(f"DPR EM Accuracy: {dpr_em_accuracy:.2f}%")
    print(f"RM3 Top-{config['top_k']} Accuracy: {rm3_accuracy_topk:.2f}%")
    print(f"RM3 EM Accuracy: {rm3_em_accuracy:.2f}%")

    return dpr_accuracy_topk, dpr_em_accuracy, rm3_accuracy_topk, rm3_em_accuracy

# ---------------主流程-----------------
def main(config, test_dataset):
    bart_generator = BartQueryGenerator(config)
    dpr_retriever = DPRRetriever(config)
    rm3_retriever = RM3Retriever(config)
    prf_expander = PRFExpander(config["rm3_index"])

    evaluate_retrieval_performance(test_dataset, dpr_retriever, rm3_retriever, bart_generator, prf_expander, config)
    

# 配置参数
config = {
    "bart_model": "facebook/bart-large",
    "output_dir": "/mnt/data/output",
    "dpr_question_encoder": "facebook/dpr-question_encoder-single-nq-base",
    "dpr_index": "/mnt/data/wikipedia-dpr-dkrr-tqa/faiss-flat.wikipedia.dkrr-dpr-tqa-retriever",
    "rm3_index": "/mnt/data/index/dataset_index_test",
    "dataset_path": "/mnt/data/ds_tq",
    "bart_max_length": 1024,
    "bart_batch_size": 256,  
    "dpr_max_length": 512,    
    "dpr_batch_size": 128,
    "expansion_terms": 5,
    "accumulation_steps": 4,
    "epochs": 3,
    "learning_rate": 1e-5,
    "top_k": 100,               # 返回前 100 个结果用于评估
    "prf_top_n": 10             # 使用前 10 篇文档进行 PRF 扩展
}

# 加载数据集
dataset = load_from_disk(config["dataset_path"])
test_dataset = dataset["test"]

if __name__ == "__main__":
    main(config, test_dataset)



