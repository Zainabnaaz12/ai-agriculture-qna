"""
Main script to set up and test the complete Project Samarth system
Run this after data collection and preprocessing
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_environment():
    """Check if all requirements are met"""
    print("🔍 Checking environment...")

    issues = []

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        issues.append("❌ ANTHROPIC_API_KEY not found in .env file")
    else:
        print("✅ Claude API key found")

    # Check data files
    data_dir = Path("data/raw")
    crop_file = data_dir / "crop_production.csv"
    rain_file = data_dir / "rainfall_data.csv"
    mandi_file = data_dir / "mandi_data.csv"  # ✅ NEW check

    if not crop_file.exists():
        issues.append(f"❌ Crop production data not found at {crop_file}")
    else:
        print(f"✅ Crop data found: {crop_file}")

    if not rain_file.exists():
        issues.append(f"❌ Rainfall data not found at {rain_file}")
    else:
        print(f"✅ Rainfall data found: {rain_file}")

    # ✅ NEW: Mandi data check
    if mandi_file.exists():
        print(f"✅ Mandi data found: {mandi_file}")
    else:
        print(f"⚠️  Optional: Mandi dataset not found at {mandi_file}")

    # Check processed data
    processed_dir = Path("data/processed")
    if not processed_dir.exists() or not list(processed_dir.glob("*.csv")):
        issues.append("⚠️  Processed data not found. Run data_preprocessing.py first")
    else:
        print("✅ Processed data found")

    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"  {issue}")
        return False

    print("\n✅ Environment check passed!\n")
    return True


def setup_system():
    """Complete system setup"""
    print("🚀 Setting up Project Samarth...\n")

    # Step 1: Check environment
    if not check_environment():
        print("\n❌ Please fix the issues above before continuing.")
        print("\nQuick fixes:")
        print("1. Create .env file with: ANTHROPIC_API_KEY=your_key_here")
        print("2. Download data and place in data/raw/")
        print("3. Run: python data_preprocessing.py")
        return False

    # Step 2: Import modules
    try:
        print("📦 Loading modules...")
        from rag_system import SamarthRAG  # ✅ corrected name
        from query_router import QueryRouter
        print("✅ Modules loaded\n")
    except ImportError as e:
        print(f"❌ Error importing modules: {e}")
        print("Run: pip install -r requirements.txt")
        return False

    # Step 3: Initialize RAG system
    try:
        print("🤖 Initializing RAG system...")
        rag = SamarthRAG()
        print("✅ RAG system initialized\n")
    except Exception as e:
        print(f"❌ Error initializing RAG: {e}")
        return False

    # Step 4: Check if vector store exists
    vector_dir = Path("vector_db")
    if not vector_dir.exists() or not list(vector_dir.glob("*")):
        print("📊 Vector store not found. Creating embeddings...")
        print("⏳ This will take 5-10 minutes...")
        try:
            rag.create_vector_store()
            print("✅ Vector store created!\n")
        except Exception as e:
            print(f"❌ Error creating vector store: {e}")
            return False
    else:
        print("✅ Vector store already exists\n")

    return rag


def test_system(rag):
    """Test the system with sample queries"""
    print("🧪 Testing system with sample queries...\n")

    test_queries = [
        "Compare average rainfall in Punjab and Haryana for 2015-2018",
        "List top 3 rice producing districts in West Bengal in 2019",
        "What is the production trend of wheat in Madhya Pradesh over last 5 years?",
        "Compare mandi prices for maize in Madhya Pradesh vs rainfall trends"  # ✅ NEW test query
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}: {query}")
        print('='*70)

        try:
            answer = rag.answer_query(query)
            print(f"\n{answer}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")

    print("\n✅ System test complete!")


def run_interactive_mode(rag):
    """Run interactive Q&A mode"""
    print("\n" + "="*70)
    print("🌾 Project Samarth - Interactive Mode")
    print("="*70)
    print("\nAsk questions about Indian agriculture and climate data.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            query = input("\n📝 Your question: ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thanks for using Project Samarth!")
                break

            print("\n🤔 Analyzing...")
            answer = rag.answer_query(query)
            print(f"\n{answer}")

        except KeyboardInterrupt:
            print("\n\n👋 Thanks for using Project Samarth!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("🌾 Project Samarth Setup & Test")
    print("="*70 + "\n")

    # Setup
    rag = setup_system()

    if not rag:
        print("\n❌ Setup failed. Please fix the issues and try again.")
        return

    print("\n" + "="*70)
    print("✅ Setup complete!")
    print("="*70)

    # Menu
    while True:
        print("\n📋 What would you like to do?")
        print("1. Run test queries")
        print("2. Interactive Q&A mode")
        print("3. Launch web interface")
        print("4. Exit")

        choice = input("\nEnter choice (1-4): ").strip()

        if choice == '1':
            test_system(rag)

        elif choice == '2':
            run_interactive_mode(rag)

        elif choice == '3':
            print("\n🌐 Launching Streamlit interface...")
            print("Run this command in a new terminal:")
            print("\n  streamlit run app.py\n")
            print("Or press Ctrl+C and run the command above.")
            break

        elif choice == '4':
            print("\n👋 Goodbye!")
            break

        else:
            print("❌ Invalid choice. Please enter 1-4.")


if __name__ == "__main__":
    main()
