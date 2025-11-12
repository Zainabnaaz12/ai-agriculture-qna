"""
Automatic Dataset Harvester for Project Samarth
Fetches and merges state-wise and district-wise crop production and rainfall data from data.gov.in
"""

import requests
import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime
import time
from dotenv import load_dotenv

load_dotenv()

class DatasetHarvester:
    def __init__(self, data_dir="data/raw_downloads", processed_dir="data/processed"):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.api_key = os.getenv("DATA_GOV_API_KEY")
        self.base_url = "https://api.data.gov.in/resource"
        self.catalog_url = "https://api.data.gov.in/v1/catalog"  # Updated catalog endpoint

        # Search keywords for datasets - expanded with more variations
        self.search_keywords = {
            "crop_production": [
                "crop production", "agriculture", "kharif", "rabi",
                "foodgrain", "cereal", "wheat", "rice", "maize",
                "state agriculture", "district agriculture"
            ],
            "rainfall": [
                "rainfall", "precipitation", "monsoon", "weather",
                "meteorological", "imd", "india meteorological department"
            ]
        }

        # Fallback resource IDs from metadata
        self.fallback_resources = {
            "crop_production": [
                "9ef84268-d588-465a-a308-a864a43d0070",  # District-wise crop production
                "0378729d-9846-4d87-9a3e-b0a7ee724054",  # State-wise crop production
                "5bf0255e-35ee-439b-aec6-99677586d07d"   # Foodgrain production
            ],
            "rainfall": [
                "6c05cd1b-ed59-40c2-bc31-e314f39c6971",  # District rainfall
                "d68b7c2e-32b3-42ab-8b6b-8468e1cc4c89"   # State rainfall
            ]
        }

        # Initialize data containers
        self.found_datasets = []
        self.downloaded_files = []
        self.merged_data = None

    def search_datasets(self, keywords, limit=50):
        """Search data.gov.in catalog for datasets using keywords and filter for CSV format"""
        print(f"🔍 Searching for datasets with keywords: {keywords}")

        found_datasets = []

        for keyword in keywords:
            print(f"   Searching for: '{keyword}'...")
            params = {
                "api-key": self.api_key,
                "format": "json",
                "q": keyword,  # Use q parameter for search
                "limit": limit
            }

            try:
                response = requests.get(self.catalog_url, params=params, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    records = data.get("records", [])  # Check if it's "records" or "data"

                    if not records:
                        records = data.get("data", [])  # Alternative key for records

                    for record in records:
                        # Check if dataset has CSV format
                        resources = record.get("resources", [])
                        csv_resources = [r for r in resources if r.get("format", "").lower() == "csv"]

                        if csv_resources:
                            dataset_info = {
                                "title": record.get("title", ""),
                                "description": record.get("description", ""),
                                "organization": record.get("org_title", ""),
                                "resource_id": csv_resources[0].get("id", ""),
                                "url": csv_resources[0].get("url", ""),
                                "keyword": keyword,
                                "format": "csv"
                            }
                            found_datasets.append(dataset_info)
                            print(f"      ✅ Found: {dataset_info['title']}")

                else:
                    print(f"      ❌ Search failed for '{keyword}': {response.status_code} - {response.text[:200]}")

            except Exception as e:
                print(f"      ❌ Error searching for '{keyword}': {e}")

            time.sleep(1)  # Rate limiting

        # Remove duplicates based on resource_id
        unique_datasets = []
        seen_ids = set()
        for ds in found_datasets:
            if ds["resource_id"] not in seen_ids:
                unique_datasets.append(ds)
                seen_ids.add(ds["resource_id"])

        self.found_datasets = unique_datasets
        print(f"📊 Found {len(unique_datasets)} unique CSV datasets")
        return unique_datasets

    def try_fallback_resources(self, data_type):
        """Try downloading known resource IDs as fallback"""
        print(f"🔄 Trying fallback resources for {data_type}...")

        fallback_datasets = []
        resource_ids = self.fallback_resources.get(data_type, [])

        for resource_id in resource_ids:
            print(f"   Trying resource ID: {resource_id}")

            # Test if resource exists by making a small request
            url = f"{self.base_url}/{resource_id}"
            params = {
                "api-key": self.api_key,
                "format": "json",
                "limit": 1  # Just test with 1 record
            }

            try:
                response = requests.get(url, params=params, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    records = data.get("records", [])

                    if records:
                        # Create dataset info
                        dataset_info = {
                            "title": f"Fallback {data_type} dataset - {resource_id}",
                            "description": f"Automatically discovered {data_type} data",
                            "organization": "data.gov.in",
                            "resource_id": resource_id,
                            "url": url,
                            "keyword": "fallback",
                            "format": "csv"
                        }
                        fallback_datasets.append(dataset_info)
                        print(f"      ✅ Valid fallback resource: {resource_id}")
                    else:
                        print(f"      ⚠️ Resource {resource_id} exists but no data")
                else:
                    print(f"      ❌ Resource {resource_id} not accessible: {response.status_code}")

            except Exception as e:
                print(f"      ❌ Error testing resource {resource_id}: {e}")

            time.sleep(1)

        return fallback_datasets

    def download_csv(self, dataset_info):
        """Download CSV file directly using resource_id"""
        resource_id = dataset_info["resource_id"]
        title = dataset_info["title"]

        print(f"📥 Downloading: {title}")
        print(f"🆔 Resource ID: {resource_id}")

        url = f"{self.base_url}/{resource_id}"
        params = {
            "api-key": self.api_key,
            "format": "csv",
            "limit": 10000  # Limit to prevent huge downloads
        }

        try:
            response = requests.get(url, params=params, timeout=60)

            if response.status_code == 200:
                # Check if response is actually CSV
                content_type = response.headers.get('content-type', '').lower()
                if 'csv' not in content_type and 'text' not in content_type:
                    print(f"      ⚠️ Unexpected content type: {content_type}")
                    return None

                # Check if content is not empty
                if len(response.content) < 100:  # Very small file, likely empty
                    print(f"      ⚠️ Empty or very small file, skipping")
                    return None

                # Save CSV file
                filename = f"{resource_id}.csv"
                filepath = self.data_dir / filename

                with open(filepath, 'wb') as f:
                    f.write(response.content)

                print(f"      💾 Saved to: {filepath}")

                # Validate CSV
                try:
                    df = pd.read_csv(filepath)
                    if df.empty:
                        print(f"      ⚠️ CSV file is empty, removing")
                        filepath.unlink()
                        return None

                    print(f"      ✅ Valid CSV with {len(df)} rows, {len(df.columns)} columns")
                    return filepath

                except Exception as e:
                    print(f"      ⚠️ Invalid CSV format: {e}, removing")
                    filepath.unlink()
                    return None

            else:
                print(f"      ❌ Download failed: {response.status_code}")
                return None

        except Exception as e:
            print(f"      ❌ Error downloading: {e}")
            return None

    def fetch_all_datasets(self):
        """Search for and download all relevant datasets"""
        print("\n" + "="*80)
        print("🌾 AUTOMATIC DATASET HARVESTER - PROJECT SAMARTH")
        print("="*80)

        if not self.api_key:
            print("❌ DATA_GOV_API_KEY not found in environment variables")
            return False

        # Search for crop production datasets
        print("\n🌾 SEARCHING FOR CROP PRODUCTION DATASETS")
        crop_datasets = self.search_datasets(self.search_keywords["crop_production"])

        # If no datasets found via search, try fallback resources
        if not crop_datasets:
            print("   No datasets found via search, trying fallback resources...")
            crop_datasets = self.try_fallback_resources("crop_production")

        # Search for rainfall datasets
        print("\n🌧️ SEARCHING FOR RAINFALL DATASETS")
        rainfall_datasets = self.search_datasets(self.search_keywords["rainfall"])

        # If no datasets found via search, try fallback resources
        if not rainfall_datasets:
            print("   No datasets found via search, trying fallback resources...")
            rainfall_datasets = self.try_fallback_resources("rainfall")

        all_datasets = crop_datasets + rainfall_datasets

        if not all_datasets:
            print("❌ No datasets found via search or fallback resources")
            return False

        # Download all found datasets
        print("\n📥 DOWNLOADING DATASETS")
        for dataset in all_datasets:
            filepath = self.download_csv(dataset)
            if filepath:
                self.downloaded_files.append(filepath)
            time.sleep(2)  # Rate limiting between downloads

        if not self.downloaded_files:
            print("❌ No valid datasets downloaded")
            return False

        print(f"✅ Downloaded {len(self.downloaded_files)} valid CSV files")
        return True

    def merge_datasets(self):
        """Merge all downloaded CSVs into a single master file"""
        print("\n🔗 MERGING ALL DATASETS")
        print("="*50)

        if not self.downloaded_files:
            print("❌ No files to merge")
            return None

        merged_dfs = []

        for filepath in self.downloaded_files:
            try:
                print(f"   Processing: {filepath.name}")
                df = pd.read_csv(filepath, low_memory=False)

                # Add source metadata
                df['_source_file'] = filepath.name
                df['_merged_at'] = datetime.now().isoformat()

                merged_dfs.append(df)
                print(f"      ✅ Added {len(df)} rows, {len(df.columns)} columns")

            except Exception as e:
                print(f"      ❌ Error reading {filepath.name}: {e}")
                continue

        if not merged_dfs:
            print("❌ No valid dataframes to merge")
            return None

        # Concatenate all dataframes
        print("   Concatenating datasets...")
        self.merged_data = pd.concat(merged_dfs, ignore_index=True, sort=False)

        # Save merged data
        master_path = self.processed_dir / "agriculture_master.csv"
        self.merged_data.to_csv(master_path, index=False)

        print("✅ Datasets merged successfully")
        print(f"💾 Saved master file to: {master_path}")
        print(f"📊 Total records: {len(self.merged_data)}")
        print(f"📊 Total columns: {len(self.merged_data.columns)}")

        return self.merged_data

    def validate_data_quality(self):
        """Validate the quality of fetched and merged data"""
        print("\n" + "="*50)
        print("🔍 DATA QUALITY VALIDATION")
        print("="*50)

        issues = []

        if self.merged_data is None:
            issues.append("❌ No merged data available")
            return False

        total_records = len(self.merged_data)
        if total_records == 0:
            issues.append("❌ No records in merged data")
        else:
            print(f"✅ Total records: {total_records}")
            print(f"📊 Columns: {list(self.merged_data.columns)}")

            # Check for key agricultural columns
            key_cols = ['state', 'district', 'crop', 'production', 'area', 'rainfall', 'year']
            found_cols = [col for col in key_cols if col in self.merged_data.columns]
            print(f"🔑 Key agricultural columns found: {found_cols}")

            # Check data completeness for found columns
            for col in found_cols:
                missing = self.merged_data[col].isnull().sum()
                if missing > 0:
                    pct = (missing / total_records) * 100
                    print(f"   {col}: {total_records - missing}/{total_records} ({100-pct:.1f}% complete)")

            # Check for state-wise data
            state_cols = ['state', 'state_name', 'State_Name']
            state_col = None
            for col in state_cols:
                if col in self.merged_data.columns:
                    state_col = col
                    break

            if state_col:
                unique_states = self.merged_data[state_col].nunique()
                print(f"📍 States covered: {unique_states}")
            else:
                issues.append("⚠️ No state column found")

            # Check year range
            year_cols = ['year', 'Year', 'crop_year']
            year_col = None
            for col in year_cols:
                if col in self.merged_data.columns:
                    year_col = col
                    break

            if year_col:
                years = pd.to_numeric(self.merged_data[year_col], errors='coerce')
                valid_years = years.dropna()
                if len(valid_years) > 0:
                    year_range = f"{int(valid_years.min())} - {int(valid_years.max())}"
                    print(f"📅 Year range: {year_range}")

        if issues:
            print("\n⚠️ Data Quality Issues:")
            for issue in issues:
                print(f"  {issue}")
            return len([i for i in issues if "❌" in i]) == 0
        else:
            print("\n✅ Data quality validation passed!")
            return True

    def run_harvester(self):
        """Run the complete harvesting process"""
        print("\n🚀 STARTING AUTOMATIC DATASET HARVESTER")
        print("="*80)

        # Step 1: Fetch all datasets
        success = self.fetch_all_datasets()
        if not success:
            print("❌ Failed to fetch datasets")
            return False

        # Step 2: Merge datasets
        merged = self.merge_datasets()
        if merged is None:
            print("❌ Failed to merge datasets")
            return False

        # Step 3: Validate data quality
        quality_ok = self.validate_data_quality()
        if not quality_ok:
            print("⚠️ Data quality issues detected - please review")

        print("\n" + "="*80)
        print("✅ HARVESTING COMPLETE")
        print("="*80)
        print("📁 Data saved to: data/raw_downloads/ and data/processed/")
        print("🔄 Next: Run data_preprocessing.py to clean the data")
        print("🤖 Then: Run rag_system.py to test Q&A capabilities")

        return True


def main():
    """Main execution function"""
    harvester = DatasetHarvester()

    # Check for API key
    if not harvester.api_key:
        print("❌ DATA_GOV_API_KEY not found in .env file")
        print("Get your API key from: https://data.gov.in/")
        print("Add it to your .env file as: DATA_GOV_API_KEY=your_key_here")
        return

    # Run the harvester
    success = harvester.run_harvester()

    if success:
        print("\n🎉 Dataset harvesting completed successfully!")
        print("Your agricultural Q&A system is now ready with comprehensive data.")
    else:
        print("\n❌ Dataset harvesting failed. Please check the errors above.")


if __name__ == "__main__":
    main()
