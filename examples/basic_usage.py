#!/usr/bin/env python3
"""
Basic usage example for GPRF Query Expansion

This example demonstrates how to use the GPRF library for query expansion
using both generative models (BART) and pseudo-relevance feedback (PRF).
"""

from gprf import BartQueryGenerator, DPRRetriever, PRFExpander
from gprf.utils.config import load_config


def main():
    """Main function demonstrating GPRF usage."""

    print("🚀 GPRF Query Expansion - Basic Usage Example")
    print("=" * 50)

    try:
        # 1. Load configuration
        print("📁 Loading configuration...")
        config = load_config("configs/default.yaml")
        print("✅ Configuration loaded successfully")

        # Note: In a real scenario, you would need trained models and indexes
        print("\n⚠️  Note: This example requires pre-trained models and indexes.")
        print("   For demonstration purposes, we'll show the API usage.")

        # 2. Initialize components (commented out for demo)
        print("\n🔧 Initializing components...")
        """
        generator = BartQueryGenerator(config)
        retriever = DPRRetriever(config)
        expander = PRFExpander(config["paths"]["index_path"])
        print("✅ All components initialized")
        """

        # 3. Prepare sample query
        print("\n📝 Preparing sample query...")
        example = {
            "Question": "What is artificial intelligence?",
            "Answer": "AI is technology that mimics human intelligence",
            "Title": "AI Overview",
            "Sentence": "Artificial intelligence refers to computer systems..."
        }
        print(f"📋 Question: {example['Question']}")

        # 4. Demonstrate query expansion (commented out for demo)
        print("\n🧠 Generating query expansions...")
        """
        # Generate BART expansions
        expansions = generator.generate_expansion_batch([example])
        print(f"🎯 BART expansions: {expansions[0]}")

        # Generate PRF expansions
        prf_terms = expander.get_prf_terms(example["Question"])
        print(f"🔍 PRF terms: {prf_terms}")

        # Combine expansions
        combined_query = construct_final_query(
            example["Question"],
            expansions[0].split(),
            prf_terms
        )
        print(f"📊 Combined query: {combined_query}")
        """

        print("✅ Query expansion simulation completed")
        print("\n💡 To run this example with real models:")
        print("   1. Train or download pre-trained BART model")
        print("   2. Build or download DPR indexes")
        print("   3. Ensure all paths in config are correct")
        print("   4. Uncomment the code blocks above")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("   - Check if all dependencies are installed")
        print("   - Verify config file exists and is valid")
        print("   - Ensure model files and indexes are available")


if __name__ == "__main__":
    main()
