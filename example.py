#!/usr/bin/env python3
"""
Example usage of the Multi-Agent RAG system.

This script demonstrates how to:
1. Create sample documents
2. Ingest them into domain-specific vector stores
3. Query the system using the router

Usage:
    python example.py
"""

import os
import tempfile
from pathlib import Path

def create_sample_documents(base_dir: Path) -> None:
    """Create sample documents for testing."""
    
    # Legal documents
    legal_dir = base_dir / "legal"
    legal_dir.mkdir(exist_ok=True)
    
    (legal_dir / "contract_law.txt").write_text("""
    Contract Law Basics
    
    A contract is a legally binding agreement between two or more parties.
    For a contract to be valid, it must have:
    1. Offer and acceptance
    2. Consideration (something of value exchanged)
    3. Legal capacity of parties
    4. Legal purpose
    
    Breach of contract occurs when one party fails to perform any duty specified in the contract.
    """)
    
    (legal_dir / "intellectual_property.txt").write_text("""
    Intellectual Property Rights
    
    Intellectual property includes:
    - Patents: Protect inventions and discoveries
    - Trademarks: Protect brand names and logos
    - Copyrights: Protect creative works
    - Trade secrets: Protect confidential business information
    
    These rights give creators exclusive use of their intellectual creations.
    """)
    
    # Code documentation
    code_dir = base_dir / "code"
    code_dir.mkdir(exist_ok=True)
    
    (code_dir / "python_basics.txt").write_text("""
    Python Programming Basics
    
    Python is a high-level, interpreted programming language.
    
    Basic data types:
    - int: Integer numbers (1, 2, 3)
    - float: Decimal numbers (1.5, 2.7)
    - str: Text strings ("hello", "world")
    - bool: True/False values
    - list: Ordered collections [1, 2, 3]
    - dict: Key-value pairs {"key": "value"}
    
    Python uses indentation to define code blocks.
    """)
    
    (code_dir / "algorithms.txt").write_text("""
    Common Algorithms
    
    Binary Search:
    def binary_search(arr, target):
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1
    
    Time complexity: O(log n)
    Space complexity: O(1)
    """)
    
    # Finance documents
    finance_dir = base_dir / "finance"
    finance_dir.mkdir(exist_ok=True)
    
    (finance_dir / "investment_basics.txt").write_text("""
    Investment Fundamentals
    
    Key investment types:
    - Stocks: Ownership shares in companies
    - Bonds: Debt securities issued by governments or corporations
    - Mutual funds: Pooled investment vehicles
    - ETFs: Exchange-traded funds
    - Real estate: Property investments
    
    Risk and return are generally correlated - higher potential returns
    typically come with higher risk.
    """)
    
    (finance_dir / "financial_ratios.txt").write_text("""
    Important Financial Ratios
    
    Profitability ratios:
    - ROE (Return on Equity) = Net Income / Shareholders' Equity
    - ROA (Return on Assets) = Net Income / Total Assets
    - Gross Margin = (Revenue - COGS) / Revenue
    
    Liquidity ratios:
    - Current Ratio = Current Assets / Current Liabilities
    - Quick Ratio = (Current Assets - Inventory) / Current Liabilities
    
    These ratios help evaluate company performance and financial health.
    """)


def main():
    """Main example function."""
    print("🚀 Multi-Agent RAG System Example")
    print("=" * 40)
    
    # Create temporary directory for sample documents
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"📁 Creating sample documents in: {temp_path}")
        create_sample_documents(temp_path)
        
        # Import and use the ingest module
        try:
            import ingest
            print("\n📚 Ingesting documents into vector stores...")
            
            # Ingest documents for each domain
            domains = ["legal", "code", "finance"]
            for domain in domains:
                domain_path = temp_path / domain
                if domain_path.exists():
                    print(f"\n   Processing {domain} documents...")
                    try:
                        ingest.ingest(domain, domain_path)
                        print(f"   ✅ Successfully ingested {domain} documents")
                    except Exception as e:
                        print(f"   ❌ Error ingesting {domain} documents: {e}")
            
            print("\n🤖 Testing query routing...")
            
            # Set a dummy model path to avoid LLM initialization
            os.environ["LLAMA_MODEL_PATH"] = "/non/existent/model/for/demo"
            
            import router
            
            # Test questions for each domain
            test_questions = [
                ("What is a contract?", "legal"),
                ("How do I implement binary search in Python?", "code"),
                ("What is ROE in finance?", "finance"),
            ]
            
            for question, expected_domain in test_questions:
                print(f"\n❓ Question: {question}")
                print(f"   Expected domain: {expected_domain}")
                
                # Since we don't have a real model, we'll get an error
                # but we can show the structure
                response, domain = router.answer(question)
                print(f"   Response: {response}")
                print(f"   Domain used: {domain}")
                
        except ImportError as e:
            print(f"❌ Error importing modules: {e}")
        except Exception as e:
            print(f"❌ Error during execution: {e}")
    
    print("\n" + "=" * 40)
    print("📖 Example completed!")
    print("\nTo use this system with a real model:")
    print("1. Download a compatible GGUF model")
    print("2. Set LLAMA_MODEL_PATH environment variable")
    print("3. Run the ingestion and query scripts")
    print("\nSee README.md for detailed instructions.")


if __name__ == "__main__":
    main()