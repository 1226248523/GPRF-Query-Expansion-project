from pyserini.search.lucene import LuceneSearcher
from pyserini.analysis import Analyzer, get_lucene_analyzer
import json
import time
import logging

logger = logging.getLogger(__name__)

# ---------------PRF Expander------------------
class PRFExpander:
    def __init__(self, index_path, config=None):
        # 使用 LuceneSearcher
        self.searcher = LuceneSearcher(index_path)
        self.searcher.set_bm25(k1=0.9, b=0.4)  # 设置 BM25 参数

        # 默认配置
        default_config = {
            'retrieval': {
                'expansion_terms': 5,
                'prf_top_n': 10
            }
        }
        self.config = config or default_config

        # 启用 RM3 查询扩展模型
        expansion_terms = self.config.get('retrieval', {}).get('expansion_terms', 5)
        self.searcher.set_rm3(
            fb_terms=expansion_terms,           # 扩展词数量
            fb_docs=10,                         # 使用前 10 篇文档作为初检结果
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
        prf_top_n = self.config.get('retrieval', {}).get('prf_top_n', 10)
        expansion_terms = self.config.get('retrieval', {}).get('expansion_terms', 5)

        hits = self.safe_search(query, prf_top_n)  # 获取前N篇文档

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
        )[:expansion_terms]

        return expanded_terms