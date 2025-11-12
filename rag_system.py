


import os
import pandas as pd
from pathlib import Path
from groq import Groq
import json
import requests
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ------------------- Configuration -------------------
DATA_DIR = Path("data/processed")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Try multiple possible filenames
CROP_FILES = [
    DATA_DIR / "crop_production_clean.csv",
    DATA_DIR / "statewise_crop_production_clean.csv",
    DATA_DIR / "foodgrain_production_clean.csv"
]
RAIN_FILE = DATA_DIR / "rainfall_clean.csv"
MANDI_FILE = DATA_DIR / "mandi_data_clean.csv"

# API Configuration
DATA_GOV_BASE_URL = "https://api.data.gov.in/resource"
DATA_GOV_API_KEY = os.getenv("DATA_GOV_API_KEY")

class SamarthRAG:
    def __init__(self):
        """Initialize RAG system with Groq API"""
        print("\n================= 🧭 Project Samarth RAG System =================")
        print("🚀 Initializing Groq-powered RAG system...")

        # Initialize Groq client
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("❌ GROQ_API_KEY not found in environment variables")

        self.client = Groq(api_key=api_key)
        # Using Llama 3.3 70B - Latest and best model
        self.model = "llama-3.3-70b-versatile"
        print(f"✅ Groq API initialized (Model: {self.model})")

        # Load or fetch datasets
        self._initialize_datasets()

        print("✅ All datasets loaded successfully!")

    def _initialize_datasets(self):
        """Initialize datasets - fetch from API if not available locally"""
        # Load metadata for API fetching
        self.metadata = self._load_metadata()

        # Try to load local data first
        self.crop_df = self._load_crop_data()
        self.foodgrain_df = self._load_foodgrain_data()
        self.rainfall_df = self._load_data(RAIN_FILE, "Rainfall")
        self.mandi_df = self._load_data(MANDI_FILE, "Mandi/Market")

        # Fetch missing datasets from API
        if self.crop_df.empty and 'crop_production' in self.metadata:
            print("📡 Fetching crop production data from data.gov.in...")
            self.crop_df = self._fetch_dataset_from_api('crop_production')

        if self.rainfall_df.empty and 'rainfall' in self.metadata:
            print("📡 Fetching rainfall data from data.gov.in...")
            self.rainfall_df = self._fetch_dataset_from_api('rainfall')

        if self.mandi_df.empty and 'mandi_prices' in self.metadata:
            print("📡 Fetching mandi price data from data.gov.in...")
            self.mandi_df = self._fetch_dataset_from_api('mandi_prices')

        # Handle foodgrain data (subset of crop production)
        if self.foodgrain_df.empty and not self.crop_df.empty:
            # Extract foodgrain data from crop data if available
            foodgrain_crops = ['Rice', 'Wheat', 'Maize', 'Bajra', 'Jowar', 'Ragi', 'Barley', 'Total Foodgrain']
            if 'crop' in self.crop_df.columns:
                self.foodgrain_df = self.crop_df[self.crop_df['crop'].str.lower().isin([c.lower() for c in foodgrain_crops])]
                if not self.foodgrain_df.empty:
                    print(f"✅ Extracted foodgrain data: {len(self.foodgrain_df)} records")

    def _load_metadata(self) -> Dict:
        """Load dataset metadata from sources_metadata.json"""
        metadata_file = Path("sources_metadata.json")
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def _fetch_dataset_from_api(self, dataset_key: str) -> pd.DataFrame:
        """Fetch dataset from data.gov.in API using resource ID"""
        if not DATA_GOV_API_KEY:
            print(f"⚠️ DATA_GOV_API_KEY not found. Cannot fetch {dataset_key} from API.")
            return pd.DataFrame()

        if dataset_key not in self.metadata:
            print(f"⚠️ No metadata found for {dataset_key}")
            return pd.DataFrame()

        resource_id = self.metadata[dataset_key]['resource_id']
        url = f"{DATA_GOV_BASE_URL}/{resource_id}"

        try:
            # First, get total records count
            params = {
                'api-key': DATA_GOV_API_KEY,
                'format': 'json',
                'limit': 1  # Just get count
            }

            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            total_records = data.get('total', 0)
            if total_records == 0:
                print(f"⚠️ No records found for {dataset_key}")
                return pd.DataFrame()

            # Fetch all records (API might have limits, so fetch in batches)
            all_records = []
            limit = 1000  # API limit per request
            offset = 0

            print(f"📡 Fetching {total_records} records for {dataset_key}...")

            while offset < total_records:
                params = {
                    'api-key': DATA_GOV_API_KEY,
                    'format': 'json',
                    'limit': limit,
                    'offset': offset
                }

                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                records = data.get('records', [])
                if not records:
                    break

                all_records.extend(records)
                offset += limit

                print(f"📡 Fetched {len(all_records)}/{total_records} records...")

                # Safety limit to prevent excessive fetching
                if len(all_records) >= 50000:
                    print("⚠️ Reached safety limit of 50,000 records")
                    break

            if not all_records:
                print(f"⚠️ No records fetched for {dataset_key}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(all_records)

            # Clean column names (remove extra spaces, standardize)
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

            # Save to CSV for caching
            output_file = DATA_DIR / f"{dataset_key}_fetched.csv"
            df.to_csv(output_file, index=False)
            print(f"✅ Saved {len(df)} records to {output_file}")

            return df

        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed for {dataset_key}: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ Error processing {dataset_key}: {e}")
            return pd.DataFrame()
    
    def _load_crop_data(self) -> pd.DataFrame:
        """Load crop data from available files"""
        for crop_file in CROP_FILES:
            if crop_file.exists():
                df = pd.read_csv(crop_file)
                print(f"✅ Loaded Crop Production: {len(df)} records from {crop_file.name}")
                return df
        print("⚠️ Crop Production data not found")
        return pd.DataFrame()

    def _load_foodgrain_data(self) -> pd.DataFrame:
        """Load foodgrain production data specifically"""
        foodgrain_file = DATA_DIR / "foodgrain_production_clean.csv"
        if foodgrain_file.exists():
            df = pd.read_csv(foodgrain_file)
            print(f"✅ Loaded Foodgrain Production: {len(df)} records")
            return df
        return pd.DataFrame()
    
    def _load_data(self, path: Path, name: str) -> pd.DataFrame:
        """Load CSV data with error handling"""
        if not path.exists():
            print(f"⚠️ {name} not found at {path}")
            return pd.DataFrame()

        df = pd.read_csv(path)
        print(f"✅ Loaded {name}: {len(df)} records")
        return df

    def _get_relevant_data(self, query: str) -> Dict[str, str]:
        """Extract relevant data based on query keywords"""
        query_lower = query.lower()
        contexts = {}
        
        # Detect crop-related queries
        if any(word in query_lower for word in ['crop', 'production', 'yield', 'wheat', 'rice', 'cotton', 'maize', 'grain', 'harvest']):
            contexts['crop'] = self._summarize_crop_data(query)
        
        # Detect rainfall-related queries
        if any(word in query_lower for word in ['rainfall', 'rain', 'precipitation', 'monsoon', 'climate', 'weather']):
            contexts['rainfall'] = self._summarize_rainfall_data(query)
        
        # Detect mandi/price queries
        if any(word in query_lower for word in ['price', 'mandi', 'market', 'rate', 'cost', 'selling']):
            contexts['mandi'] = self._summarize_mandi_data(query)
        
        # If no keywords matched, include all datasets
        if not contexts:
            contexts['crop'] = self._summarize_crop_data(query)
            contexts['rainfall'] = self._summarize_rainfall_data(query)
            contexts['mandi'] = self._summarize_mandi_data(query)
        
        return contexts
    
    def _summarize_crop_data(self, query: str) -> str:
        """Create focused summary of crop data with source citations"""
        # First check foodgrain data for specific crop queries
        if not self.foodgrain_df.empty and 'rice' in query.lower():
            rice_data = self.foodgrain_df[self.foodgrain_df['crop'].str.lower() == 'rice']
            if not rice_data.empty:
                production = rice_data['production__lakh_tons____2023_24'].iloc[0]
                productivity = rice_data['productivity__yield_in_kg_ha____2023_24'].iloc[0]
                return f"Total Rice Production in India for 2023-24: {production} lakh tons\nRice Productivity: {productivity} kg/ha"

        # Check for top crops query
        if 'top' in query.lower() and ('crop' in query.lower() or 'production' in query.lower()):
            if not self.foodgrain_df.empty:
                # Filter out total rows and get individual crops
                individual_crops = self.foodgrain_df[~self.foodgrain_df['crop'].str.contains('Total', case=False, na=False)]
                if not individual_crops.empty:
                    # Sort by production and get top 5
                    top_crops = individual_crops.nlargest(5, 'production__lakh_tons____2023_24')[['crop', 'production__lakh_tons____2023_24', 'productivity__yield_in_kg_ha____2023_24']]
                    top_crops_str = "\n".join([f"{row['crop']}: {row['production__lakh_tons____2023_24']} lakh tons (Yield: {row['productivity__yield_in_kg_ha____2023_24']} kg/ha)" for _, row in top_crops.iterrows()])
                    return f"Top 5 crops by production in India (2023-24):\n{top_crops_str}"

        if self.crop_df.empty:
            return "Crop production data not available."

        # Extract states, crops from query
        states = self._extract_entities(query, self.crop_df, 'state_name')
        crops = self._extract_entities(query, self.crop_df, 'crop')

        df = self.crop_df.copy()

        # Filter by states if mentioned
        if states:
            df = df[df['state_name'].str.lower().isin([s.lower() for s in states])]

        # Filter by crops if mentioned
        if crops:
            df = df[df['crop'].str.lower().isin([c.lower() for c in crops])]

        # Limit to recent years
        if 'year' in df.columns:
            df = df[df['year'] >= 2010]

        # Limit data size to avoid token limits
        df = df.head(100)

        summary_parts = []

        if not df.empty:
            # Top producing states
            if 'production' in df.columns and 'state_name' in df.columns:
                top_states = df.groupby('state_name')['production'].sum().nlargest(10)
                summary_parts.append(f"Top producing states:\n{top_states.to_dict()}")

            # Top crops
            if 'crop' in df.columns and 'production' in df.columns:
                top_crops = df.groupby('crop')['production'].sum().nlargest(10)
                summary_parts.append(f"\nTop crops by production:\n{top_crops.to_dict()}")

            # Sample records with key columns
            key_cols = ['state_name', 'district_name', 'crop', 'year', 'production', 'area', 'yield']
            available_cols = [col for col in key_cols if col in df.columns]
            sample = df[available_cols].tail(20).to_dict('records')
            summary_parts.append(f"\nSample records:\n{json.dumps(sample, indent=2, default=str)}")

        return "\n".join(summary_parts) if summary_parts else "No matching crop data found."
    
    def _summarize_rainfall_data(self, query: str) -> str:
        """Create focused summary of rainfall data with source citations"""
        if self.rainfall_df.empty:
            return "Rainfall data not available."

        # Extract states and districts from query
        states = self._extract_entities(query, self.rainfall_df, 'state_name')
        districts = self._extract_entities(query, self.rainfall_df, 'district_name')

        df = self.rainfall_df.copy()

        # Filter by states if mentioned
        if states:
            df = df[df['state_name'].str.lower().isin([s.lower() for s in states])]

        # Filter by districts if mentioned
        if districts:
            df = df[df['district_name'].str.lower().isin([d.lower() for d in districts])]

        # Filter by recent years
        if 'year' in df.columns:
            df = df[df['year'] >= 2018]  # NRSC data starts from 2018

        # Limit data size for processing
        df = df.head(1000)  # Increased for variability calculations

        summary_parts = []

        if not df.empty:
            # Check for variability queries
            variability_keywords = ['variability', 'variation', 'deviation', 'coefficient', 'cv', 'standard deviation', 'std']
            is_variability_query = any(word in query.lower() for word in variability_keywords)

            if is_variability_query:
                # Calculate rainfall variability (coefficient of variation)
                if 'avg_rainfall' in df.columns and 'state_name' in df.columns:
                    # Group by state and calculate CV
                    state_stats = df.groupby('state_name')['avg_rainfall'].agg(['mean', 'std']).dropna()
                    state_stats['cv'] = (state_stats['std'] / state_stats['mean']) * 100  # Coefficient of variation as percentage

                    # Sort by highest variability
                    highest_variability = state_stats.nlargest(10, 'cv')[['mean', 'std', 'cv']]
                    summary_parts.append(f"States with highest rainfall variability (Coefficient of Variation %):\n{highest_variability.to_dict('index')}")

                    # Also show lowest variability for comparison
                    lowest_variability = state_stats.nsmallest(5, 'cv')[['mean', 'std', 'cv']]
                    summary_parts.append(f"\nStates with lowest rainfall variability (most consistent):\n{lowest_variability.to_dict('index')}")

                # District-level variability if districts mentioned or for detailed analysis
                if 'district_name' in df.columns and 'avg_rainfall' in df.columns:
                    district_stats = df.groupby('district_name')['avg_rainfall'].agg(['mean', 'std']).dropna()
                    district_stats['cv'] = (district_stats['std'] / district_stats['mean']) * 100

                    highest_district_variability = district_stats.nlargest(10, 'cv')[['mean', 'std', 'cv']]
                    summary_parts.append(f"\nDistricts with highest rainfall variability:\n{highest_district_variability.to_dict('index')}")

            else:
                # Standard rainfall summaries
                # Average rainfall by state
                if 'avg_rainfall' in df.columns and 'state_name' in df.columns:
                    avg_rain = df.groupby('state_name')['avg_rainfall'].mean().nlargest(10)
                    summary_parts.append(f"Average daily rainfall by state (mm):\n{avg_rain.to_dict()}")

                # District-level rainfall if districts mentioned
                if districts and 'district_name' in df.columns and 'avg_rainfall' in df.columns:
                    district_rain = df.groupby('district_name')['avg_rainfall'].mean().nlargest(10)
                    summary_parts.append(f"\nDistrict-wise average rainfall (mm):\n{district_rain.to_dict()}")

                # Monthly patterns if available
                if 'month' in df.columns and 'avg_rainfall' in df.columns:
                    monthly_avg = df.groupby('month')['avg_rainfall'].mean()
                    summary_parts.append(f"\nAverage rainfall by month (mm):\n{monthly_avg.to_dict()}")

            # Sample records with key columns
            key_cols = ['state_name', 'district_name', 'date', 'year', 'month', 'avg_rainfall', 'agency_name']
            available_cols = [col for col in key_cols if col in df.columns]
            sample = df[available_cols].tail(20).to_dict('records')
            summary_parts.append(f"\nSample district-wise rainfall records:\n{json.dumps(sample, indent=2, default=str)}")

        return "\n".join(summary_parts) if summary_parts else "No matching rainfall data found."
    
    def _summarize_mandi_data(self, query: str) -> str:
        """Create focused summary of mandi price data with source citations"""
        if self.mandi_df.empty:
            return "Mandi price data not available."

        df = self.mandi_df.copy()
        states = self._extract_entities(query, df, 'state')
        crops = self._extract_entities(query, df, 'commodity')

        if states:
            df = df[df['state'].str.lower().isin([s.lower() for s in states])]

        if crops:
            df = df[df['commodity'].str.lower().isin([c.lower() for c in crops])]

        df = df.head(100)

        summary_parts = []

        if not df.empty:
            if 'modal_price' in df.columns and 'state' in df.columns and 'commodity' in df.columns:
                avg_prices = df.groupby(['state', 'commodity'])['modal_price'].mean().nlargest(10)
                summary_parts.append(f"Average mandi prices:\n{avg_prices.to_dict()}")

            key_cols = ['state', 'district', 'market', 'commodity', 'modal_price', 'min_price', 'max_price', 'year', 'month']
            available_cols = [col for col in key_cols if col in df.columns]
            sample = df[available_cols].tail(15).to_dict('records')
            summary_parts.append(f"\nSample price records:\n{json.dumps(sample, indent=2, default=str)}")

        return "\n".join(summary_parts) if summary_parts else "No matching mandi data found."
    
    def _extract_entities(self, query: str, df: pd.DataFrame, column: str) -> List[str]:
        """Extract entities mentioned in query"""
        if column not in df.columns:
            return []
        
        query_lower = query.lower()
        entities = []
        
        for value in df[column].dropna().unique():
            if str(value).lower() in query_lower:
                entities.append(str(value))
        
        return entities[:5]
    
    def answer_query(self, query: str) -> str:
        """Generate answer using Groq API with retrieved context"""
        print(f"\n🔍 Processing query: {query}")
        
        # Get relevant data
        contexts = self._get_relevant_data(query)
        
        # Build context string
        context_str = "\n\n".join([f"=== {source.upper()} DATA (from data.gov.in) ===\n{content}" 
                                   for source, content in contexts.items()])
        
        if not context_str:
            context_str = "No relevant data found in the databases."
        
        # Create prompt
        prompt = f"""You are an expert agricultural data analyst for the Indian government. You have access to official datasets from data.gov.in including:

1. **Crop Production Data** - Ministry of Agriculture & Farmers Welfare
2. **Rainfall Data** - India Meteorological Department (IMD)
3. **Mandi Price Data** - Agricultural Marketing Department

USER QUESTION: {query}

RELEVANT DATA RETRIEVED FROM DATASETS:
{context_str}

INSTRUCTIONS:
1. Analyze the data carefully and provide a comprehensive, accurate answer
2. Include specific numbers, statistics, and comparisons from the data
3. **IMPORTANT**: For EVERY claim or data point, cite the source dataset explicitly (e.g., "According to the Crop Production Data..." or "The Rainfall Data shows...")
4. If comparing multiple entities (states, crops, years), present clear side-by-side comparisons
5. If trends are requested, describe patterns over time with specific years and values
6. Structure your answer with clear sections if needed
7. If data is insufficient for a complete answer, acknowledge what's missing but provide all available insights
8. Be precise and quantitative - use actual numbers from the data
9. If asked for policy recommendations, base them strictly on the data patterns you observe

Provide your detailed analysis now:"""

        try:
            # Call Groq API
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert agricultural data analyst specializing in Indian agriculture. You always cite your sources and provide accurate, data-driven insights."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.3,
                max_tokens=2000,
                top_p=0.9
            )
            
            answer = chat_completion.choices[0].message.content
            print("\n✅ Answer generated successfully!")
            return answer
            
        except Exception as e:
            error_msg = f"Error generating answer: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg


# ------------------- Main Execution -------------------
if __name__ == "__main__":
    # Initialize system
    rag = SamarthRAG()
    
    # Interactive mode
    print("\n" + "="*70)
    print("🌾 Project Samarth - Interactive Q&A")
    print("="*70)
    print("\nType 'quit' to exit\n")
    
    # Check if running in non-interactive mode (e.g., piped input)
    import sys
    if not sys.stdin.isatty():
        # Read from stdin (piped input)
        query = sys.stdin.read().strip()
        if query:
            answer = rag.answer_query(query)
            print(f"\n🤖 Answer:\n{answer}\n")
        sys.exit(0)

    # Interactive mode
    while True:
        try:
            query = input("💬 Your question: ").strip()
        except EOFError:
            # Handle EOF (e.g., when piped input ends)
            break

        if not query:
            continue

        if query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thank you for using Project Samarth!")
            break

        answer = rag.answer_query(query)
        print(f"\n🤖 Answer:\n{answer}\n")
        print("-" * 70)
