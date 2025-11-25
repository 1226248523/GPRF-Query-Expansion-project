"""
运行模型评估的脚本
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gprf.utils.config import load_config
from gprf.utils.evaluation import evaluate_retrieval_performance
from gprf.core import BartQueryGenerator, DPRRetriever, RM3Retriever, PRFExpander

def main():
    # 加载配置
    config = load_config("configs/default.yaml")
    
    # 初始化组件
    bart_generator = BartQueryGenerator(config)
    dpr_retriever = DPRRetriever(config)
    rm3_retriever = RM3Retriever(config)
    index_path = config.get("paths", {}).get("index_dir", "indexes")
    prf_expander = PRFExpander(index_path, config)
    
    # 运行评估
    evaluate_retrieval_performance(
        test_dataset=None,  
        dpr_retriever=dpr_retriever,
        rm3_retriever=rm3_retriever,
        bart_generator=bart_generator,
        prf_expander=prf_expander,
        config=config
    )

if __name__ == "__main__":
    main()