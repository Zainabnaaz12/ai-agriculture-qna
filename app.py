"""
Streamlit UI for Project Samarth
Interactive Q&A interface powered by Groq API
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import plotly.express as px
from rag_system import SamarthRAG

# Page configuration
st.set_page_config(
    page_title="Project Samarth - Agricultural Intelligence",
    page_icon="🌾",
    layout="wide"
)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🌾 Project Samarth</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Agricultural Intelligence System</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("About Project Samarth")
    st.info("""
    An intelligent Q&A system analyzing government datasets from **data.gov.in**:
    
    🌾 **Crop Production** - Ministry of Agriculture  
    🌧️ **Rainfall Data** - IMD  
    💰 **Mandi Prices** - Agricultural Marketing  
    
    **Powered by Groq API** ⚡ 
    """)
    
    st.markdown("---")
    
    if st.button("🔄 Initialize System", type="primary"):
        with st.spinner("Loading RAG system..."):
            try:
                st.session_state.rag_system = SamarthRAG()
                st.success("✅ System ready!")
                st.balloons()
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("💡 Make sure GROQ_API_KEY is set in your .env file")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("---")
    
    # Stats
    if st.session_state.rag_system:
        st.metric("Model", "Llama 3.1 70B")
        st.metric("Queries Asked", len(st.session_state.chat_history))
    
    st.caption("Built for Bharat Digital Internship Challenge")

# Main content
tab1, tab2, tab3 = st.tabs(["💬 Ask Questions", "📊 Data Explorer", "ℹ️ Sample Queries"])

with tab1:
    st.header("Ask Your Questions")
    
    if st.session_state.rag_system is None:
        st.warning("⚠️ Please initialize the system from the sidebar first.")
        st.info("👈 Click the **Initialize System** button to start")
    else:
        # Query input
        user_query = st.text_input(
            "Enter your question:",
            placeholder="e.g., Compare wheat production in Punjab and Haryana over last 5 years",
            key="query_input"
        )
        
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            ask_button = st.button("🔍 Ask Question", type="primary", use_container_width=True)
        with col2:
            if st.button("🎲 Random Sample", use_container_width=True):
                samples = [
                    "Which are the top 5 rice producing states?",
                    "Compare rainfall in Maharashtra and Karnataka",
                    "What is the wheat production trend in Punjab?",
                    "List top crops by production in India"
                ]
                import random
                user_query = random.choice(samples)
                ask_button = True
        
        # Quick query buttons
        st.markdown("**Quick queries:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🌾 Top rice states", use_container_width=True):
                user_query = "Which are the top 5 rice producing states in India?"
                ask_button = True
        with col2:
            if st.button("🌧️ Rainfall trends", use_container_width=True):
                user_query = "Compare rainfall trends in Maharashtra and Karnataka"
                ask_button = True
        with col3:
            if st.button("💰 Crop prices", use_container_width=True):
                user_query = "What are the average mandi prices for wheat?"
                ask_button = True
        
        if ask_button and user_query:
            with st.spinner("🤔 Analyzing data..."):
                try:
                    answer = st.session_state.rag_system.answer_query(user_query)
                    
                    # Add to history
                    st.session_state.chat_history.append({
                        "query": user_query,
                        "answer": answer
                    })
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("Check your internet connection and API key")
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("---")
            st.subheader("💬 Conversation History")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.container():
                    col1, col2 = st.columns([0.95, 0.05])
                    with col1:
                        st.markdown(f"**🙋 Question {len(st.session_state.chat_history) - i}:**")
                    with col2:
                        if st.button("🗑️", key=f"del_{i}"):
                            st.session_state.chat_history.pop(len(st.session_state.chat_history) - 1 - i)
                            st.rerun()
                    
                    st.info(chat['query'])
                    st.markdown(f"**🤖 Answer:**")
                    st.success(chat['answer'])
                    st.markdown("---")

with tab2:
    st.header("📊 Data Explorer")
    
    if st.session_state.rag_system is not None:
        rag = st.session_state.rag_system
        
        data_type = st.selectbox(
            "Select Dataset", 
            ["Crop Production", "Rainfall Data", "Mandi Prices"],
            help="Choose which dataset to explore"
        )
        
        if data_type == "Crop Production" and not rag.crop_df.empty:
            df = rag.crop_df
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📊 Total Records", f"{len(df):,}")
            col2.metric("🗺️ States", df['state_name'].nunique() if 'state_name' in df.columns else 0)
            col3.metric("🌾 Crops", df['crop'].nunique() if 'crop' in df.columns else 0)
            col4.metric("📅 Years", f"{df['year'].min()}-{df['year'].max()}" if 'year' in df.columns else "N/A")
            
            # Filters
            with st.expander("🔍 Filter Data"):
                col1, col2 = st.columns(2)
                with col1:
                    if 'state_name' in df.columns:
                        states = st.multiselect("States", df['state_name'].unique())
                        if states:
                            df = df[df['state_name'].isin(states)]
                with col2:
                    if 'crop' in df.columns:
                        crops = st.multiselect("Crops", df['crop'].unique())
                        if crops:
                            df = df[df['crop'].isin(crops)]
            
            # Display
            st.subheader("📋 Sample Data")
            st.dataframe(df.head(100), use_container_width=True)
            st.caption(f"Showing {min(100, len(df))} of {len(df):,} records")
            
        elif data_type == "Rainfall Data" and not rag.rainfall_df.empty:
            df = rag.rainfall_df
            
            col1, col2, col3 = st.columns(3)
            col1.metric("📊 Total Records", f"{len(df):,}")
            col2.metric("🗺️ Regions", df['subdivision'].nunique() if 'subdivision' in df.columns else 0)
            col3.metric("📅 Years", f"{df['year'].min()}-{df['year'].max()}" if 'year' in df.columns else "N/A")
            
            st.subheader("📋 Sample Data")
            st.dataframe(df.head(100), use_container_width=True)
            st.caption(f"Showing {min(100, len(df))} of {len(df):,} records")
            
        elif data_type == "Mandi Prices" and not rag.mandi_df.empty:
            df = rag.mandi_df
            
            col1, col2, col3 = st.columns(3)
            col1.metric("📊 Total Records", f"{len(df):,}")
            col2.metric("🗺️ States", df['state'].nunique() if 'state' in df.columns else 0)
            col3.metric("🌾 Commodities", df['commodity'].nunique() if 'commodity' in df.columns else 0)
            
            st.subheader("📋 Sample Data")
            st.dataframe(df.head(100), use_container_width=True)
            st.caption(f"Showing {min(100, len(df))} of {len(df):,} records")
        else:
            st.warning("⚠️ Dataset not available or empty")
    else:
        st.warning("⚠️ Please initialize the system first.")

with tab3:
    st.header("💡 Sample Queries")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌾 Production Queries")
        st.markdown("""
        - Which are the top 5 rice producing states in India?
        - Compare wheat production between Punjab and Haryana over the last 5 years
        - List the most produced crops in Maharashtra
        - Which districts in Punjab had highest wheat production in 2020?
        - What is the total production of cereals in India?
        """)
        
        st.subheader("🌧️ Climate Queries")
        st.markdown("""
        - Compare average rainfall in Maharashtra and Karnataka
        - Analyze rainfall trends in Gujarat over the past decade
        - Which states experienced the highest rainfall variability?
        - What is the monsoon pattern in West Bengal?
        """)
    
    with col2:
        st.subheader("💰 Market Queries")
        st.markdown("""
        - What are the average mandi prices for cotton?
        - Compare wheat prices across Punjab and Haryana
        - Show price trends for rice in major markets
        """)
        
        st.subheader("📊 Analysis Queries")
        st.markdown("""
        - Correlate rainfall patterns with rice production in West Bengal
        - How does rainfall affect crop yield in drought-prone regions?
        - Analyze the relationship between monsoon and sugarcane production
        - Identify water-intensive crops in low-rainfall regions
        """)
    
    st.markdown("---")
    
    st.subheader("🎯 Policy Questions")
    st.markdown("""
    - Based on data, what are three arguments to promote millets in Karnataka?
    - Which states show declining rainfall but stable crop production?
    - Identify regions suitable for drought-resistant crops
    - What crops should be promoted in high-rainfall zones?
    """)
    
    st.success("""
    💡 **Tips for Better Results:**
    
    ✅ Be specific about states, districts, crops, and time periods  
    ✅ Ask for comparisons to get detailed analysis  
    ✅ Request trends to see patterns over time  
    ✅ Groq's Llama 3.1 70B provides fast, accurate answers with citations!  
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🌾 <strong>Project Samarth</strong> - Agricultural Intelligence System</p>
    <p>Powered by <strong>Groq API</strong> (Llama 3.1 70B) | Data from <a href='https://data.gov.in' target='_blank'>data.gov.in</a></p>
    <p><em>Bharat Digital Fellowship Challenge 2026</em></p>
</div>
""", unsafe_allow_html=True)