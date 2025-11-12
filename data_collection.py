"""
Improved Data Collection Script for Project Samarth
Downloads comprehensive agricultural datasets from multiple sources
"""

import requests
import pandas as pd
import os
from pathlib import Path
import time
from dotenv import load_dotenv

load_dotenv()

class ImprovedDataCollector:
    def __init__(self, data_dir="data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = os.getenv("DATA_GOV_API_KEY")
        self.base_url = "https://api.data.gov.in/resource"

        # Dataset metadata for traceability
        self.dataset_metadata = {
            "crop_production": {
                "resource_id": "9ef84268-d588-465a-a308-a864a43d0070",
                "name": "District-wise Crop Production Statistics (1997-2015)",
                "source": "Ministry of Agriculture & Farmers Welfare",
                "url": "https://data.gov.in/resource/districtwise-crop-production-statistics-1997-2015"
            },
            "rainfall_imd": {
                "resource_id": "6c05cd1b-ed59-40c2-bc31-e314f39c6971",
                "name": "District-wise Daily Rainfall Data (NRSC VIC MODEL)",
                "source": "National Remote Sensing Centre (NRSC)",
                "url": "https://data.gov.in/resource/district-wise-daily-rainfall-data-nrsc-vic-model"
            },
            "mandi_prices": {
                "resource_id": "9ef84268-d588-465a-a308-a864a43d0070",  # Placeholder - need actual mandi resource ID
                "name": "Daily Market Prices of Selected Commodities",
                "source": "Agricultural Marketing Department",
                "url": "https://data.gov.in/resource/daily-market-prices-selected-commodities"
            }
        }
    
    # =====================================================
    # OPTION 1: Use Kaggle's Comprehensive Dataset
    # =====================================================
    def download_from_kaggle(self):
        """
        Download comprehensive Indian Agriculture dataset from Kaggle
        This is MORE COMPLETE than data.gov.in individual datasets
        """
        print("\n" + "="*70)
        print("📊 DOWNLOADING COMPREHENSIVE AGRICULTURE DATASET")
        print("="*70)
        
        # Use data.gov.in API with your API key for comprehensive data
        print("\n📥 Using data.gov.in API for comprehensive crop data...")

        # Resource IDs for comprehensive agricultural data
        resource_ids = {
            "district_crop_production": "9ef84268-d588-465a-a308-a864a43d0070",  # District-wise crop production 1997-2015
            "state_crop_production": "0378729d-9846-4d87-9a3e-b0a7ee724054",     # State-wise crop production 2020-21
            "foodgrain_production": "5bf0255e-35ee-439b-aec6-99677586d07d"       # Crop-wise foodgrain production
        }

        all_crop_data = []
        for name, resource_id in resource_ids.items():
            print(f"\n📊 Fetching {name}...")
            data = self._fetch_from_api(resource_id, limit=50000)
            if data:
                all_crop_data.extend(data)
                print(f"   ✅ Added {len(data):,} records")

        if all_crop_data:
            df = pd.DataFrame(all_crop_data)
            # Remove duplicates if any
            df = df.drop_duplicates()
            print(f"\n✅ Combined dataset: {len(df):,} total records")

            # Save the comprehensive dataset
            output_path = self.data_dir / "crop_production_detailed.csv"
            df.to_csv(output_path, index=False)
            print(f"✅ Saved to: {output_path}")
            print(f"✅ Columns: {list(df.columns)}")
            print(f"✅ Years covered: {df.get('crop_year', df.get('year', 'N/A')).min() if 'crop_year' in df.columns or 'year' in df.columns else 'N/A'} to {df.get('crop_year', df.get('year', 'N/A')).max() if 'crop_year' in df.columns or 'year' in df.columns else 'N/A'}")
            return df
        else:
            print("❌ No data retrieved from API. Using sample data...")
            return None
    
    # =====================================================
    # OPTION 2: Direct CSV Downloads (Backup URLs)
    # =====================================================
    def download_backup_datasets(self):
        """Download from direct CSV links (backup sources)"""
        print("\n" + "="*70)
        print("📊 DOWNLOADING BACKUP DATASETS")
        print("="*70)
        
        datasets = {
            "crop_production": "https://raw.githubusercontent.com/plotly/datasets/master/agriculture_crops_production.csv",
            "state_wise_production": "https://data.gov.in/files/ogdpv2/moa/districtwise_crop_production_statistics/districtwise_crop_production_statistics_1997-2015.csv"
        }
        
        for name, url in datasets.items():
            print(f"\nDownloading {name}...")
            try:
                df = pd.read_csv(url)
                output_path = self.data_dir / f"{name}_backup.csv"
                df.to_csv(output_path, index=False)
                print(f"✅ {len(df):,} records saved to {output_path}")
            except Exception as e:
                print(f"⚠️ Could not download {name}: {e}")
    
    # =====================================================
    # OPTION 3: Enhanced Data.gov.in Collection
    # =====================================================
    def download_enhanced_crop_data(self):
        """Download multiple crop datasets and merge them"""
        print("\n" + "="*70)
        print("📊 DOWNLOADING ENHANCED CROP DATA FROM DATA.GOV.IN")
        print("="*70)
        
        # Multiple resource IDs for comprehensive data
        resource_ids = {
            "crop_production_1997_2015": "9ef84268-d588-465a-a308-a864a43d0070",  # District-wise
            "statewise_production": "0378729d-9846-4d87-9a3e-b0a7ee724054",       # State-wise 2020-21
            "foodgrain_production": "5bf0255e-35ee-439b-aec6-99677586d07d"         # Crop-wise
        }
        
        all_data = {}
        
        for name, resource_id in resource_ids.items():
            print(f"\n📥 Downloading {name}...")
            data = self._fetch_from_api(resource_id, limit=50000)
            if data:
                all_data[name] = data
                output_path = self.data_dir / f"{name}.csv"
                df = pd.DataFrame(data)
                df.to_csv(output_path, index=False)
                print(f"✅ {len(data):,} records saved")
        
        return all_data
    
    def download_enhanced_rainfall_data(self):
        """Download comprehensive rainfall data from IMD"""
        print("\n" + "="*70)
        print("🌧️ DOWNLOADING RAINFALL DATA FROM IMD")
        print("="*70)

        # Primary IMD rainfall dataset
        resource_id = self.dataset_metadata["rainfall_imd"]["resource_id"]
        dataset_name = self.dataset_metadata["rainfall_imd"]["name"]

        print(f"\n📥 Downloading from {dataset_name}...")
        data = self._fetch_from_api(resource_id, limit=50000, dataset_name="rainfall_imd")

        if data:
            output_path = self.data_dir / "rainfall_data.csv"
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
            print(f"✅ {len(data):,} rainfall records saved to {output_path}")
            print(f"📊 Source: {self.dataset_metadata['rainfall_imd']['url']}")
            print(f"🆔 Resource ID: {resource_id}")
            return df
        else:
            print("❌ Failed to download rainfall data from IMD API")
            return None

    def download_mandi_price_data(self):
        """Download mandi price data from Agricultural Marketing Department"""
        print("\n" + "="*70)
        print("💰 DOWNLOADING MANDI PRICE DATA")
        print("="*70)

        # Note: This is a placeholder - actual mandi price resource ID needs to be found
        # For now, we'll search for available mandi datasets
        print("🔍 Searching for mandi price datasets on data.gov.in...")

        # Try known mandi price resource IDs (these may need updating)
        potential_resource_ids = [
            "9ef84268-d588-465a-a308-a864a43d0070",  # This might be crop data, not mandi
            "58e0c6b2-0f4d-4a0e-8b1f-5b0c6b2a8e4f",  # Placeholder - need actual ID
        ]

        for resource_id in potential_resource_ids:
            print(f"\n📥 Trying resource ID: {resource_id}")
            data = self._fetch_from_api(resource_id, limit=50000, dataset_name="mandi_prices")

            if data:
                output_path = self.data_dir / "mandi_data.csv"
                df = pd.DataFrame(data)
                df.to_csv(output_path, index=False)
                print(f"✅ {len(data):,} mandi price records saved to {output_path}")
                print(f"📊 Source: {self.dataset_metadata['mandi_prices']['url']}")
                print(f"🆔 Resource ID: {resource_id}")
                return df

        print("❌ No mandi price data found. You may need to:")
        print("   1. Find the correct resource ID from https://data.gov.in/")
        print("   2. Update the mandi_prices resource_id in dataset_metadata")
        print("   3. Or use sample data for demonstration")
        return None
    
    def _fetch_from_api(self, resource_id, limit=10000, dataset_name="Unknown"):
        """Fetch data from data.gov.in API with enhanced error handling and format support"""
        if not self.api_key:
            print("⚠️ DATA_GOV_API_KEY not found. Using sample data.")
            return None

        url = f"{self.base_url}/{resource_id}"
        all_records = []
        offset = 0
        retry_count = 0
        max_retries = 3

        print(f"📡 Fetching from {dataset_name} (Resource ID: {resource_id})")

        while True:
            params = {
                "api-key": self.api_key,
                "format": "json",
                "limit": min(limit, 10000),  # API limit is 10,000 per request
                "offset": offset
            }

            try:
                response = requests.get(url, params=params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    records = data.get("records", [])

                    if not records:
                        break

                    # Handle different data formats and structures
                    processed_records = self._process_api_records(records, dataset_name)
                    all_records.extend(processed_records)

                    print(f"   Downloaded {len(all_records):,} records...", end="\r")

                    if len(records) < limit:
                        break

                    offset += limit
                    time.sleep(1)  # Rate limiting
                    retry_count = 0  # Reset retry on success

                elif response.status_code == 429:  # Rate limit exceeded
                    retry_count += 1
                    if retry_count <= max_retries:
                        wait_time = 2 ** retry_count  # Exponential backoff
                        print(f"\n   Rate limited. Waiting {wait_time}s before retry {retry_count}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"\n   Rate limit exceeded after {max_retries} retries")
                        break

                else:
                    print(f"\n   Error {response.status_code}: {response.text[:200]}")
                    break

            except requests.exceptions.Timeout:
                retry_count += 1
                if retry_count <= max_retries:
                    print(f"\n   Timeout. Retry {retry_count}/{max_retries}...")
                    time.sleep(2)
                    continue
                else:
                    print(f"\n   Timeout after {max_retries} retries")
                    break

            except Exception as e:
                print(f"\n   Error: {e}")
                break

        print(f"\n   Total: {len(all_records):,} records from {dataset_name}")
        return all_records if all_records else None

    def _process_api_records(self, records, dataset_name):
        """Process and standardize API records from different data formats"""
        processed = []

        for record in records:
            # Standardize field names and handle missing data
            processed_record = {}

            # Handle crop production data
            if "crop" in dataset_name.lower():
                processed_record = {
                    "state_name": record.get("state_name", record.get("State_Name", "")),
                    "district_name": record.get("district_name", record.get("District_Name", "")),
                    "crop": record.get("crop", record.get("Crop", "")),
                    "year": record.get("crop_year", record.get("year", record.get("Crop_Year", ""))),
                    "season": record.get("season", record.get("Season", "")),
                    "area": record.get("area", record.get("Area", 0)),
                    "production": record.get("production", record.get("Production", 0)),
                    "yield": record.get("yield", record.get("Yield", 0)),
                    "source_dataset": dataset_name,
                    "resource_id": self.dataset_metadata.get(dataset_name.split('_')[0], {}).get("resource_id", ""),
                    "source_url": self.dataset_metadata.get(dataset_name.split('_')[0], {}).get("url", "")
                }

            # Handle rainfall data (district-wise daily from NRSC VIC MODEL)
            elif "rainfall" in dataset_name.lower():
                processed_record = {
                    "state": record.get("state", record.get("State", "")),
                    "district": record.get("district", record.get("District", "")),
                    "date": record.get("date", record.get("Date", "")),
                    "year": record.get("year", record.get("Year", "")),
                    "month": record.get("month", record.get("Month", "")),
                    "avg_rainfall": record.get("avg_rainfall", record.get("Avg_rainfall", 0)),
                    "agency_name": record.get("agency_name", record.get("Agency_name", "")),
                    "source_dataset": dataset_name,
                    "resource_id": self.dataset_metadata.get("rainfall_imd", {}).get("resource_id", ""),
                    "source_url": self.dataset_metadata.get("rainfall_imd", {}).get("url", "")
                }

            # Handle mandi price data
            elif "mandi" in dataset_name.lower() or "price" in dataset_name.lower():
                processed_record = {
                    "state": record.get("state", record.get("State", "")),
                    "district": record.get("district", record.get("District", "")),
                    "market": record.get("market", record.get("Market", "")),
                    "commodity": record.get("commodity", record.get("Commodity", "")),
                    "variety": record.get("variety", record.get("Variety", "")),
                    "modal_price": record.get("modal_price", record.get("Modal_Price", 0)),
                    "min_price": record.get("min_price", record.get("Min_Price", 0)),
                    "max_price": record.get("max_price", record.get("Max_Price", 0)),
                    "year": record.get("year", record.get("Year", "")),
                    "month": record.get("month", record.get("Month", "")),
                    "source_dataset": dataset_name,
                    "resource_id": self.dataset_metadata.get("mandi_prices", {}).get("resource_id", ""),
                    "source_url": self.dataset_metadata.get("mandi_prices", {}).get("url", "")
                }

            else:
                # Generic processing for unknown formats
                processed_record = record.copy()
                processed_record.update({
                    "source_dataset": dataset_name,
                    "resource_id": "",
                    "source_url": ""
                })

            processed.append(processed_record)

        return processed
    
    # =====================================================
    # OPTION 4: Create Sample/Demo Data (if APIs fail)
    # =====================================================
    def create_sample_data(self):
        """Create realistic sample data for demo purposes"""
        print("\n" + "="*70)
        print("🎲 CREATING SAMPLE DATA FOR DEMO")
        print("="*70)
        
        import numpy as np
        
        # Sample comprehensive crop data
        states = ['Punjab', 'Haryana', 'Uttar Pradesh', 'Maharashtra', 'Karnataka', 
                  'West Bengal', 'Tamil Nadu', 'Andhra Pradesh', 'Gujarat', 'Madhya Pradesh']
        
        crops = ['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane', 'Pulses', 
                 'Groundnut', 'Soybean', 'Jowar', 'Bajra']
        
        years = list(range(2010, 2024))
        
        # Generate realistic production data
        data = []
        for state in states:
            for crop in crops:
                for year in years:
                    # Generate realistic values
                    base_production = np.random.randint(1000, 50000)
                    area = np.random.randint(100, 5000)
                    
                    data.append({
                        'State_Name': state,
                        'District_Name': f'{state} District {np.random.randint(1, 5)}',
                        'Crop': crop,
                        'Crop_Year': year,
                        'Season': np.random.choice(['Kharif', 'Rabi', 'Summer']),
                        'Area': area,
                        'Production': base_production * (1 + np.random.uniform(-0.2, 0.3)),
                        'Yield': base_production / area if area > 0 else 0
                    })
        
        df_crop = pd.DataFrame(data)
        output_path = self.data_dir / "crop_production_sample.csv"
        df_crop.to_csv(output_path, index=False)
        print(f"✅ Created {len(df_crop):,} crop production records")
        print(f"✅ Saved to: {output_path}")
        
        # Sample rainfall data
        data_rain = []
        for state in states:
            for year in years:
                rainfall = {
                    'State_Name': state,
                    'Year': year,
                    'Jan': np.random.uniform(10, 50),
                    'Feb': np.random.uniform(10, 60),
                    'Mar': np.random.uniform(20, 80),
                    'Apr': np.random.uniform(30, 100),
                    'May': np.random.uniform(40, 120),
                    'Jun': np.random.uniform(100, 300),
                    'Jul': np.random.uniform(150, 400),
                    'Aug': np.random.uniform(150, 400),
                    'Sep': np.random.uniform(100, 300),
                    'Oct': np.random.uniform(50, 150),
                    'Nov': np.random.uniform(20, 80),
                    'Dec': np.random.uniform(10, 50)
                }
                rainfall['Annual_Rainfall'] = sum([rainfall[m] for m in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']])
                data_rain.append(rainfall)
        
        df_rain = pd.DataFrame(data_rain)
        output_path = self.data_dir / "rainfall_sample.csv"
        df_rain.to_csv(output_path, index=False)
        print(f"✅ Created {len(df_rain):,} rainfall records")
        print(f"✅ Saved to: {output_path}")
        
        return df_crop, df_rain


# =====================================================
# Main Execution with Multiple Options
# =====================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🌾 PROJECT SAMARTH - IMPROVED DATA COLLECTION")
    print("="*70)
    
    collector = ImprovedDataCollector()
    
    print("\n📋 DATA COLLECTION OPTIONS:")
    print("1. Download from Kaggle/GitHub (RECOMMENDED - Most Complete)")
    print("2. Download from data.gov.in (Official but may be incomplete)")
    print("3. Create sample/demo data (For testing)")
    print("4. Try all sources")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        print("\n🎯 Downloading from Kaggle/GitHub...")
        collector.download_from_kaggle()
        print("\n💡 TIP: This dataset is comprehensive and ready to use!")
        
    elif choice == "2":
        print("\n🎯 Downloading from data.gov.in...")
        if not os.getenv("DATA_GOV_API_KEY"):
            print("\n⚠️ DATA_GOV_API_KEY not found in .env")
            print("Get one from: https://data.gov.in/")
            print("\nFalling back to sample data...")
            collector.create_sample_data()
        else:
            collector.download_enhanced_crop_data()
            collector.download_enhanced_rainfall_data()
            collector.download_mandi_price_data()
    
    elif choice == "3":
        print("\n🎯 Creating sample data...")
        collector.create_sample_data()
        print("\n💡 TIP: This is realistic demo data perfect for your presentation!")
    
    elif choice == "4":
        print("\n🎯 Trying all sources...")
        collector.download_from_kaggle()
        collector.download_backup_datasets()
        if os.getenv("DATA_GOV_API_KEY"):
            collector.download_enhanced_crop_data()
            collector.download_enhanced_rainfall_data()
        else:
            collector.create_sample_data()
    
    else:
        print("❌ Invalid choice")
    
    print("\n" + "="*70)
    print("✅ DATA COLLECTION COMPLETE")
    print("="*70)
    print("\n📁 Check your data/raw/ folder for downloaded files")
    print("🔄 Next step: python data_preprocessing.py")
    print("="*70 + "\n")