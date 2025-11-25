import pytest
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Common test fixtures can be defined here
@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "model": {
            "bart_model": "facebook/bart-base",
            "dpr_encoder": "facebook/dpr-question_encoder-single-nq-base"
        },
        "training": {
            "batch_size": 2,
            "max_length": 512,
            "epochs": 1
        },
        "retrieval": {
            "top_k": 10,
            "expansion_terms": 3,
            "prf_top_n": 5
        },
        "paths": {
            "data_dir": "data",
            "model_dir": "models",
            "index_dir": "indexes"
        }
    }

@pytest.fixture
def sample_example():
    """Sample query example for testing."""
    return {
        "Question": "What is artificial intelligence?",
        "Answer": "AI is technology that mimics human intelligence",
        "Title": "AI Overview",
        "Sentence": "Artificial intelligence refers to computer systems..."
    }
