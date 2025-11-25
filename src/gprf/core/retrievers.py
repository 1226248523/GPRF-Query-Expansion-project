import torch
from pyserini.search.faiss import FaissSearcher
from pyserini.search.lucene import LuceneSearcher

# ---------------DPR Retriever-----------------
class DPRRetriever:
    def __init__(self, config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # 初始化 DPR 查询编码器
        encoder_name = config.get('model', {}).get('dpr_encoder', 'facebook/dpr-question_encoder-single-nq-base')
        self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(encoder_name)
        self.model = DPRQuestionEncoder.from_pretrained(encoder_name).to(self.device)

        # 加载 Faiss 索引目录
        self.index_dir = config.get('paths', {}).get('index_dir', 'indexes')
        self.searcher = FaissSearcher(self.index_dir)

        # 配置参数
        self.max_length = config.get('training', {}).get('max_length', 512)
        self.batch_size = config.get('training', {}).get('batch_size', 128)

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
        self.index_path = config.get('paths', {}).get('index_dir', 'indexes')
        self.searcher = LuceneSearcher(self.index_path)

    def retrieve(self, queries, top_k=100):
        results = []
        for query in queries:
            hits = self.searcher.search(query, k=top_k)
            results.append(hits)
        return results