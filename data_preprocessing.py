"""
Data Preprocessing Script for Project Samarth
Cleans and standardizes agriculture and climate data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

class DataPreprocessor:
    def __init__(self, raw_dir="data/raw", processed_dir="data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def preprocess_crop_data(self):
        """Clean and standardize crop production data"""
        print("Preprocessing crop production data...")
        
        # Try both possible file names
        possible_files = [
            "statewise_crop_production_2020_21.csv",
            "cropwise_foodgrain_production.csv"
        ]
        
        df = None
        for filename in possible_files:
            filepath = self.raw_dir / filename
            if filepath.exists():
                print(f"Found: {filename}")
                df = pd.read_csv(filepath)
                break
        
        if df is None:
            print("❌ No crop production file found!")
            return pd.DataFrame()
        
        print(f"Original shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Map common column variations to standard names
        column_mapping = {
            'state': 'state_name',
            'states': 'state_name',
            'state_name': 'state_name',
            'district': 'district_name',
            'districts': 'district_name',
            'district_name': 'district_name',
            'crop': 'crop',
            'crops': 'crop',
            'crop_name': 'crop',
            'season': 'season',
            'year': 'year',
            'crop_year': 'year',
            'area': 'area',
            'production': 'production',
            'yield': 'yield'
        }
        
        # Rename columns based on mapping
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)
        
        # Remove rows with missing critical values
        critical_cols = []
        if 'state_name' in df.columns:
            critical_cols.append('state_name')
        if 'crop' in df.columns:
            critical_cols.append('crop')
        
        if critical_cols:
            df = df.dropna(subset=critical_cols)
        
        # Clean and standardize text fields
        text_cols = ['state_name', 'district_name', 'crop', 'season']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.title()
        
        # Convert numeric columns
        numeric_cols = ['area', 'production', 'year']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate yield if not present
        if 'yield' not in df.columns and 'area' in df.columns and 'production' in df.columns:
            df['yield'] = df['production'] / df['area']
            df['yield'] = df['yield'].replace([np.inf, -np.inf], np.nan)
        
        # Create crop categories
        if 'crop' in df.columns:
            df['crop_category'] = df['crop'].apply(self._categorize_crop)
        
        # Remove obvious outliers (negative values, etc.)
        for col in ['area', 'production', 'yield']:
            if col in df.columns:
                df = df[df[col] >= 0]
        
        print(f"Processed shape: {df.shape}")
        
        # Save processed data
        output_path = self.processed_dir / "crop_production_clean.csv"
        df.to_csv(output_path, index=False)
        print(f"✅ Saved to {output_path}")
        
        # Create metadata
        self._save_metadata(df, "crop_production_metadata.json")
        
        return df
    
    def preprocess_rainfall_data(self):
        """Clean and standardize rainfall data"""
        print("\nPreprocessing rainfall data...")
        
        filepath = self.raw_dir / "rainfall_data.csv"
        
        if not filepath.exists():
            print(f"❌ Rainfall data not found at {filepath}")
            return pd.DataFrame()
        
        df = pd.read_csv(filepath)
        
        print(f"Original shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Map common variations
        column_mapping = {
            'state': 'state_name',
            'state_name': 'state_name',
            'state_ut_name': 'state_name',
            'subdivision': 'subdivision',
            'sub_division': 'subdivision',
            'district': 'district_name',
            'year': 'year',
            'annual': 'annual_rainfall',
            'annual_rainfall': 'annual_rainfall'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)
        
        # Clean text fields
        text_cols = ['state_name', 'subdivision', 'district_name']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.title()
        
        # Convert month columns to numeric
        month_cols = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                      'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
                      'january', 'february', 'march', 'april', 'may', 'june',
                      'july', 'august', 'september', 'october', 'november', 'december']
        
        for col in month_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate annual rainfall if not present
        if 'annual_rainfall' not in df.columns:
            available_months = [col for col in month_cols if col in df.columns]
            if available_months:
                df['annual_rainfall'] = df[available_months].sum(axis=1)
                print(f"✅ Calculated annual rainfall from {len(available_months)} months")
        
        # Convert year
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
        
        # Remove negative values
        if 'annual_rainfall' in df.columns:
            df = df[df['annual_rainfall'] >= 0]
        
        print(f"Processed shape: {df.shape}")
        
        # Save processed data
        output_path = self.processed_dir / "rainfall_clean.csv"
        df.to_csv(output_path, index=False)
        print(f"✅ Saved to {output_path}")
        
        # Create metadata
        self._save_metadata(df, "rainfall_metadata.json")
        
        return df

    def preprocess_foodgrain_data(self):
        """Clean and standardize foodgrain production data"""
        print("\nPreprocessing foodgrain data...")
        
        filepath = self.raw_dir / "cropwise_foodgrain_production.csv"
        
        if not filepath.exists():
            print(f"⚠️ Foodgrain data not found at {filepath}")
            return pd.DataFrame()
        
        df = pd.read_csv(filepath)
        
        print(f"Original shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Save as additional dataset
        output_path = self.processed_dir / "foodgrain_production_clean.csv"
        df.to_csv(output_path, index=False)
        print(f"✅ Saved to {output_path}")
        
        return df

    def preprocess_mandi_data(self):
        """Clean and standardize mandi price data (if available)"""
        print("\nPreprocessing mandi price data...")

        filepath = self.raw_dir / "mandi_data.csv"
        
        if not filepath.exists():
            print(f"⚠️ Mandi data not found at {filepath} (optional)")
            return pd.DataFrame()

        df = pd.read_csv(filepath)

        print(f"Original shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

        # Drop missing values in key columns
        key_cols = ['state', 'district', 'market', 'commodity', 'arrival_date']
        existing_key_cols = [col for col in key_cols if col in df.columns]
        if existing_key_cols:
            df = df.dropna(subset=existing_key_cols)

        # Clean text columns
        text_cols = ['state', 'district', 'market', 'commodity', 'variety']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.title()

        # Convert numeric columns
        numeric_cols = ['min_price', 'max_price', 'modal_price', 'arrival_quantity']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert date
        if 'arrival_date' in df.columns:
            df['arrival_date'] = pd.to_datetime(df['arrival_date'], errors='coerce')

        # Extract useful features
        if 'arrival_date' in df.columns:
            df['year'] = df['arrival_date'].dt.year
            df['month'] = df['arrival_date'].dt.month
            df['day'] = df['arrival_date'].dt.day

        # Save processed file
        output_path = self.processed_dir / "mandi_data_clean.csv"
        df.to_csv(output_path, index=False)
        print(f"✅ Saved to {output_path}")

        # Save metadata
        self._save_metadata(df, "mandi_metadata.json")

        return df

    def create_state_mapping(self, crop_df, rainfall_df):
        """Create mapping between crop states and rainfall subdivisions"""
        print("\nCreating state-subdivision mapping...")
        
        mapping = {}
        
        if not crop_df.empty and not rainfall_df.empty:
            state_col = 'state_name' if 'state_name' in crop_df.columns else None
            rain_col = 'subdivision' if 'subdivision' in rainfall_df.columns else 'state_name'
            
            if state_col and rain_col in rainfall_df.columns:
                states = crop_df[state_col].unique()
                subdivisions = rainfall_df[rain_col].unique()
                
                for state in states:
                    matches = [sub for sub in subdivisions 
                              if str(state).lower() in str(sub).lower() or 
                              str(sub).lower() in str(state).lower()]
                    mapping[state] = matches[0] if matches else None
        
        mapping_path = self.processed_dir / "state_subdivision_mapping.json"
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        print(f"✅ Saved mapping to {mapping_path}")
        return mapping
    
    def _categorize_crop(self, crop_name):
        """Categorize crops into types"""
        if pd.isna(crop_name):
            return "Unknown"
        
        crop_lower = str(crop_name).lower()
        
        cereals = ['rice', 'wheat', 'maize', 'bajra', 'jowar', 'barley', 'ragi', 'paddy']
        if any(c in crop_lower for c in cereals):
            return "Cereal"
        
        pulses = ['gram', 'tur', 'moong', 'urad', 'lentil', 'peas', 'masoor', 'arhar', 'pulse']
        if any(p in crop_lower for p in pulses):
            return "Pulse"
        
        cash_crops = ['cotton', 'sugarcane', 'jute', 'tobacco']
        if any(cc in crop_lower for cc in cash_crops):
            return "Cash Crop"
        
        oilseeds = ['groundnut', 'soybean', 'sunflower', 'safflower', 'rapeseed', 'mustard', 'oilseed']
        if any(o in crop_lower for o in oilseeds):
            return "Oilseed"
        
        return "Other"
    
    def _save_metadata(self, df, filename):
        """Save dataset metadata"""
        metadata = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_summary": df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
        }
        
        output_path = self.processed_dir / filename
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"📊 Saved metadata to {output_path}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🌾 PROJECT SAMARTH - DATA PREPROCESSING")
    print("="*70 + "\n")
    
    preprocessor = DataPreprocessor()
    
    # Check what files exist
    raw_dir = Path("data/raw")
    print("📂 Checking for data files...")
    for file in raw_dir.glob("*.csv"):
        print(f"   ✅ Found: {file.name}")
    print()
    
    # Preprocess available data
    crop_df = preprocessor.preprocess_crop_data()
    rainfall_df = preprocessor.preprocess_rainfall_data()
    foodgrain_df = preprocessor.preprocess_foodgrain_data()
    mandi_df = preprocessor.preprocess_mandi_data()
    
    # Create mappings if possible
    if not crop_df.empty and not rainfall_df.empty:
        mapping = preprocessor.create_state_mapping(crop_df, rainfall_df)
    
    print("\n" + "="*70)
    print("✅ PREPROCESSING COMPLETE")
    print("="*70)
    
    if not crop_df.empty:
        print(f"✅ Crop data: {crop_df.shape[0]:,} records")
    else:
        print("⚠️  Crop data: Not processed")
    
    if not rainfall_df.empty:
        print(f"✅ Rainfall data: {rainfall_df.shape[0]:,} records")
    else:
        print("⚠️  Rainfall data: Not processed")
    
    if not foodgrain_df.empty:
        print(f"✅ Foodgrain data: {foodgrain_df.shape[0]:,} records")
    else:
        print("⚠️  Foodgrain data: Not processed")
    
    if not mandi_df.empty:
        print(f"✅ Mandi data: {mandi_df.shape[0]:,} records")
    else:
        print("⚠️  Mandi data: Optional (not found)")
    
    print("\n📁 Processed files saved to: data/processed/")
    print("\n🚀 Next step: Run 'python rag_system.py' to test the system!")
    print("="*70 + "\n")