"""
Quick test script for Project Samarth (Groq Version)
Run this to verify your system is working correctly
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from rag_system import SamarthRAG

def test_environment():
    """Test environment setup"""
    print("\n" + "="*70)
    print("🔍 TESTING ENVIRONMENT")
    print("="*70)
    
    # Load environment variables
    load_dotenv()
    
    issues = []
    
    # Check API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        issues.append("❌ GROQ_API_KEY not found in .env")
    elif not api_key.startswith("gsk_"):
        issues.append("⚠️ GROQ_API_KEY format looks incorrect (should start with 'gsk_')")
    else:
        print("✅ Groq API key found")
    
    # Check data files
    data_dir = Path("data/processed")
    required_files = [
        "crop_production_clean.csv",
        "rainfall_clean.csv"
    ]
    
    optional_files = [
        "mandi_data_clean.csv"
    ]
    
    for filename in required_files:
        filepath = data_dir / filename
        if filepath.exists():
            print(f"✅ Found: {filename}")
        else:
            issues.append(f"❌ Missing: {filename}")
    
    for filename in optional_files:
        filepath = data_dir / filename
        if filepath.exists():
            print(f"✅ Found: {filename}")
        else:
            print(f"⚠️  Optional: {filename} not found (system will work without it)")
    
    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"   {issue}")
        return False
    
    print("\n✅ Environment check passed!")
    return True


def test_rag_initialization():
    """Test RAG system initialization"""
    print("\n" + "="*70)
    print("🚀 TESTING RAG INITIALIZATION")
    print("="*70)
    
    try:
        rag = SamarthRAG()
        print("✅ RAG system initialized successfully")
        print(f"✅ Using model: {rag.model}")
        return rag
    except Exception as e:
        print(f"❌ Failed to initialize RAG: {e}")
        return None


def test_sample_queries(rag):
    """Test with sample queries"""
    print("\n" + "="*70)
    print("🧪 TESTING SAMPLE QUERIES")
    print("="*70)
    
    test_queries = [
        {
            "query": "Which are the top 3 rice producing states?",
            "expected": "Should mention states and production numbers with citations"
        },
        {
            "query": "Compare rainfall in Maharashtra and Karnataka",
            "expected": "Should compare rainfall data between states with specific numbers"
        },
        {
            "query": "What is the trend of wheat production in Punjab?",
            "expected": "Should describe trends with year-over-year data"
        }
    ]
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"Test Query {i}/{len(test_queries)}")
        print(f"{'='*70}")
        print(f"Query: {test['query']}")
        print(f"Expected: {test['expected']}\n")
        
        try:
            answer = rag.answer_query(test['query'])
            print(f"\n📝 Answer:\n{answer}\n")
            
            # Check if answer has citations
            has_citation = any(phrase in answer.lower() for phrase in [
                'according to', 'data shows', 'rainfall data', 'crop production',
                'ministry of', 'source:', 'dataset'
            ])
            
            if has_citation:
                print("✅ Query processed successfully (citations found)")
            else:
                print("⚠️  Query processed but citations may be missing")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🌾 PROJECT SAMARTH - SYSTEM TEST (GROQ VERSION)")
    print("="*70)
    
    # Step 1: Test environment
    if not test_environment():
        print("\n❌ Environment check failed. Please fix issues and try again.")
        print("\n📋 Quick fixes:")
        print("1. Get Groq API key from: https://console.groq.com/keys")
        print("2. Create .env file with: GROQ_API_KEY=gsk_your_key_here")
        print("3. Run: python data_collection.py")
        print("4. Run: python data_preprocessing.py")
        return
    
    # Step 2: Initialize RAG
    rag = test_rag_initialization()
    if not rag:
        print("\n❌ RAG initialization failed. Check your API key and internet connection.")
        return
    
    # Step 3: Test queries
    print("\n⏳ Testing queries (this will take ~30-60 seconds)...")
    if not test_sample_queries(rag):
        print("\n❌ Query testing failed.")
        return
    
    # Success!
    print("\n" + "="*70)
    print("🎉 ALL TESTS PASSED!")
    print("="*70)
    print("\n✨ Your system is ready! Next steps:")
    print("\n1. 💻 Run CLI: python rag_system.py")
    print("2. 🌐 Run Web UI: streamlit run app.py")
    print("3. 🎥 Record your Loom video")
    print("\n💡 Groq API Benefits:")
    print("   • FREE tier: 14,400 requests/day")
    print("   • Super fast responses (1-3 seconds)")
    print("   • High quality answers")
    print("\n🚀 Good luck with your submission tomorrow!")


if __name__ == "__main__":
    main()