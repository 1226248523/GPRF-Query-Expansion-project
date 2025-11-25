#!/usr/bin/env python
"""
Traditional setup.py for GPRF Query Expansion
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gprf-query-expansion",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="GPRF: Query Expansion using Generative Models and Pseudo-Relevance Feedback",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/gprf-query-expansion",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "transformers>=4.21.0",
        "datasets>=2.4.0",
        "pyserini>=0.21.0",
        "faiss-cpu>=1.7.0",
        "tqdm>=4.64.0",
        "PyYAML>=6.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "jupyter>=1.0.0",
        ],
    },
)