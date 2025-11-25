import pytest
from gprf.core.generators import BartQueryGenerator

class TestBartQueryGenerator:
    def test_initialization(self):
        """测试生成器初始化"""
        config = {"bart_model": "facebook/bart-base"}
        generator = BartQueryGenerator(config)
        assert generator is not None
    
    def test_format_input(self):
        """测试输入格式化"""
        config = {"bart_model": "facebook/bart-base"}
        generator = BartQueryGenerator(config)
        
        example = {
            "Question": "What is AI?",
            "Answer": "Artificial Intelligence",
            "Title": "AI Overview",
            "Sentence": "AI is technology."
        }
        
        result = generator.format_input(example)
        expected = "What is AI? Answer: Artificial Intelligence Title: AI Overview Sentence: AI is technology."
        assert result == expected