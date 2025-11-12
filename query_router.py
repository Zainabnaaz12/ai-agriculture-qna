# query_router.py
# -----------------------------
# Smart Query Router for Project Samarth
# Handles Crop, Rainfall, and Mandi Price datasets intelligently
# -----------------------------

import pandas as pd
from typing import Dict, List


class QueryRouter:
    def __init__(self, crop_df: pd.DataFrame, rainfall_df: pd.DataFrame, mandi_df: pd.DataFrame):
        self.crop_df = crop_df
        self.rainfall_df = rainfall_df
        self.mandi_df = mandi_df

    # --------------------------------------------------------
    # STEP 1: Analyze the query and detect intent + data sources
    # --------------------------------------------------------
    def analyze_query(self, query: str) -> Dict:
        query = query.lower()

        # Identify relevant datasets
        data_sources = self._determine_data_sources(query)

        # Extract key entities
        crops = self._extract_keywords(query, ['rice', 'wheat', 'maize', 'cotton', 'pulses', 'sugarcane'])
        states = self._extract_keywords(query, [
            'andhra pradesh', 'maharashtra', 'uttar pradesh', 'bihar', 'karnataka',
            'tamil nadu', 'gujarat', 'rajasthan', 'madhya pradesh', 'odisha', 'west bengal'
        ])
        years = [int(word) for word in query.split() if word.isdigit() and 1990 < int(word) < 2030]

        return {
            'data_sources': data_sources,
            'crops': crops,
            'states': states,
            'years': years,
            'intent': self._determine_intent(query)
        }

    # --------------------------------------------------------
    # STEP 2: Decide which datasets to use
    # --------------------------------------------------------
    def _determine_data_sources(self, query: str) -> List[str]:
        sources = []
        crop_keywords = ['crop', 'yield', 'production', 'cultivation']
        rainfall_keywords = ['rainfall', 'precipitation', 'monsoon']
        mandi_keywords = ['price', 'market', 'mandi', 'rate', 'selling price', 'modal price']

        if any(kw in query for kw in crop_keywords):
            sources.append('crop')
        if any(kw in query for kw in rainfall_keywords):
            sources.append('rainfall')
        if any(kw in query for kw in mandi_keywords):
            sources.append('mandi')

        return sources

    # --------------------------------------------------------
    # STEP 3: Detect user's question type
    # --------------------------------------------------------
    def _determine_intent(self, query: str) -> str:
        if 'average' in query or 'mean' in query:
            return 'average'
        elif 'trend' in query or 'increase' in query or 'decrease' in query:
            return 'trend'
        elif 'correlation' in query or 'relationship' in query:
            return 'correlation'
        elif 'compare' in query or 'difference' in query:
            return 'comparison'
        else:
            return 'summary'

    # --------------------------------------------------------
    # STEP 4: Execute query
    # --------------------------------------------------------
    def execute_direct_query(self, query: str) -> pd.DataFrame:
        analysis = self.analyze_query(query)
        params = {
            'crops': analysis['crops'],
            'states': analysis['states'],
            'years': analysis['years']
        }

        if 'crop' in analysis['data_sources']:
            return self._get_crop_data(params)
        elif 'rainfall' in analysis['data_sources']:
            return self._get_rainfall_data(params)
        elif 'mandi' in analysis['data_sources']:
            return self._get_mandi_prices(params)
        else:
            return pd.DataFrame({'message': ['No matching dataset found for your query.']})

    # --------------------------------------------------------
    # STEP 5A: Crop dataset logic
    # --------------------------------------------------------
    def _get_crop_data(self, params: Dict) -> pd.DataFrame:
        df = self.crop_df.copy()

        if params['states']:
            df = df[df['State_Name'].isin(params['states'])]
        if params['crops']:
            df = df[df['Crop'].str.lower().isin([c.lower() for c in params['crops']])]
        if params['years']:
            df = df[df['Crop_Year'].isin(params['years'])]

        result = df.groupby(['State_Name', 'Crop'])['Production'].sum().reset_index()
        result.rename(columns={'Production': 'Total_Production'}, inplace=True)
        return result

    # --------------------------------------------------------
    # STEP 5B: Rainfall dataset logic
    # --------------------------------------------------------
    def _get_rainfall_data(self, params: Dict) -> pd.DataFrame:
        df = self.rainfall_df.copy()

        if params['states']:
            df = df[df['STATE_UT_NAME'].isin(params['states'])]
        if params['years']:
            df = df[df['YEAR'].isin(params['years'])]

        result = df.groupby(['STATE_UT_NAME', 'YEAR'])['ANNUAL'].mean().reset_index()
        result.rename(columns={'ANNUAL': 'Avg_Annual_Rainfall'}, inplace=True)
        return result

    # --------------------------------------------------------
    # STEP 5C: Mandi dataset logic
    # --------------------------------------------------------
    def _get_mandi_prices(self, params: Dict) -> pd.DataFrame:
        df = self.mandi_df.copy()

        if params['states']:
            df = df[df['state'].isin(params['states'])]
        if params['crops']:
            df = df[df['commodity'].str.lower().isin([c.lower() for c in params['crops']])]
        if params['years']:
            df = df[df['year'].isin(params['years'])]

        result = df.groupby(['state', 'commodity'])['modal_price'].mean().reset_index()
        result.rename(columns={'modal_price': 'Avg_Modal_Price'}, inplace=True)
        return result

    # --------------------------------------------------------
    # STEP 6: Helper for keyword extraction
    # --------------------------------------------------------
    def _extract_keywords(self, query: str, keywords: List[str]) -> List[str]:
        return [kw for kw in keywords if kw in query]

    # --------------------------------------------------------
    # STEP 7: Format output neatly
    # --------------------------------------------------------
    def format_results(self, df: pd.DataFrame) -> str:
        if df.empty:
            return "No matching data found."
        return df.to_string(index=False)
