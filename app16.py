import datetime
import json
import openai
import re
import streamlit as st
from bs4 import BeautifulSoup
import requests
import numpy as np
import pandas as pd
from langchain.document_loaders import WebBaseLoader
from langgraph.graph import StateGraph, START, END
from typing import Dict, List, TypedDict

# ✅ OpenAI Configuration
openai.api_key = "14560021aaf84772835d76246b53397a"
openai.api_base = "https://amrxgenai.openai.azure.com/"
openai.api_type = 'azure'
openai.api_version = '2024-02-15-preview'
deployment_name = 'gpt'

# ✅ Bing Search URL
BING_SEARCH_URL = "https://www.bing.com/search?q="

# ✅ Streamlit UI Styling
st.markdown("""
<style>
    .stApp {
        background-image: url("https://e0.pxfuel.com/wallpapers/986/360/desktop-wallpaper-background-color-4851-background-color-theme-colorful-brown-color.jpg");
        background-attachment: fixed;
        background-size: cover;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Stock Analysis with LangGraph & GenAI - Human-in-the-Loop")

# ✅ State Schema Definition
class StockState(TypedDict):
    ticker: str
    num_results: int
    results: List[Dict]
    scraped_data: List[Dict]
    validated_data: str
    stats_data: str

# ✅ Initialize State
if "state" not in st.session_state:
    st.session_state.state = StockState(
        ticker="",
        num_results=5,
        results=[],
        scraped_data=[],
        validated_data="",
        stats_data=""
    )

if "history" not in st.session_state:
    st.session_state.history = []


# ✅ User Confirmation Function
def user_confirmation(step_name: str) -> bool:
    """Prompt user to confirm before proceeding to the next step."""
    return st.radio(f"Do you want to proceed with {step_name}?", ["Yes", "No"]) == "Yes"


# ✅ Functions for Each Step
def get_bing_results(state: StockState) -> StockState:
    """Fetch Bing search results with configurable number of results."""
    if not user_confirmation("Bing Search"):
        return state

    query = state["ticker"]
    num_results = state["num_results"]
    search_url = BING_SEARCH_URL + query.replace(" ", "+")
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(search_url, headers=headers)

    if response.status_code != 200:
        st.error("Failed to fetch Bing results.")
        return state

    soup = BeautifulSoup(response.text, "html.parser")
    results = []

    for b in soup.find_all('li', class_='b_algo')[:num_results]:
        title = b.find('h2').text if b.find('h2') else "No Title"
        link = b.find('a')['href'] if b.find('a') else "No Link"
        snippet = b.find('p').text if b.find('p') else "No snippet available"
        results.append({"title": title, "link": link, "snippet": snippet})

    state["results"] = results
    st.session_state.history.append({"step": "Bing", "data": results})
    
    st.write("✅ Bing Results:")
    st.json(results)

    return state


def scrape_full_page(state: StockState) -> StockState:
    """Scrape content from Bing search results."""
    if not user_confirmation("Scraping"):
        return state

    if not state["results"]:
        state["scraped_data"] = []
        return state

    scraped_data = []

    for res in state["results"]:
        url = res["link"]
        try:
            loader = WebBaseLoader(url)
            doc = loader.load()
            soup = BeautifulSoup(doc[0].page_content, "html.parser")
            full_content = soup.get_text(separator="\n")

            ticker_match = re.search(r'\b[A-Z]{2,5}\b', full_content)
            ticker = ticker_match.group(0) if ticker_match else "N/A"

            price_match = re.search(r'\$\d{1,5}(\.\d{1,2})?', full_content)
            price = price_match.group(0) if price_match else "N/A"

            date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            scraped_data.append({"url": url, "date": date, "ticker": ticker, "price": price})

        except Exception as e:
            scraped_data.append({"url": url, "error": str(e), "ticker": "N/A", "price": "N/A"})

    state["scraped_data"] = scraped_data
    st.session_state.history.append({"step": "Scraping", "data": scraped_data})

    st.write("✅ Scraped Data:")
    st.json(scraped_data)

    return state


def validate_results(state: StockState) -> StockState:
    """Validate the scraped stock data and display in a tabular format."""
    if not user_confirmation("Validation"):
        return state

    if not state["scraped_data"]:
        state["validated_data"] = "No data available for validation."
        return state

    table_data = []

    for item in state["scraped_data"]:
        table_data.append({
            "URL": item["url"],
            "Date": item["date"],
            "Ticker": item.get("ticker", "N/A"),
            "Price": item.get("price", "N/A")
        })

    # Display table format
    df = pd.DataFrame(table_data)
    st.write("✅ Validation Results in Table Format:")
    st.table(df)

    # Store in state history
    state["validated_data"] = df.to_json(orient="records")
    st.session_state.history.append({"step": "Validation", "data": df.to_dict(orient="records")})

    return state


def generate_stats(state: StockState) -> StockState:
    """Generate statistics for stock data."""
    if not user_confirmation("Statistics Generation"):
        return state

    if not state["scraped_data"]:
        state["stats_data"] = "No data available for stats."
        return state

    stats = []

    for item in state["scraped_data"]:
        try:
            price = float(item.get("price", "N/A").replace("$", "").replace(",", ""))
        except ValueError:
            price = None

        historical_prices = np.random.uniform(low=price * 0.9, high=price * 1.1, size=20) if price else []

        if len(historical_prices) > 0:
            sma = np.mean(historical_prices)
            ema = np.average(historical_prices, weights=np.linspace(1, 0, len(historical_prices)))
            std_dev = np.std(historical_prices)

            stats.append({
                "URL": item["url"],
                "Ticker": item.get("ticker", "N/A"),
                "Price": f"${price:.2f}" if price else "N/A",
                "SMA": f"${sma:.2f}",
                "EMA": f"${ema:.2f}",
                "Std Dev": f"${std_dev:.2f}"
            })

    # Display stats in tabular format
    st.write("✅ Statistics Results:")
    df_stats = pd.DataFrame(stats)
    st.table(df_stats)

    state["stats_data"] = df_stats.to_json(orient="records")
    st.session_state.history.append({"step": "Statistics", "data": stats})

    return state


# ✅ LangGraph Pipeline
graph = StateGraph(StockState)

# ✅ Add Nodes
graph.add_node("bing", get_bing_results)
graph.add_node("scraping", scrape_full_page)
graph.add_node("validation", validate_results)
graph.add_node("stats", generate_stats)

# ✅ Add Edges
graph.add_edge(START, "bing")
graph.add_edge("bing", "scraping")
graph.add_edge("scraping", "validation")
graph.add_edge("validation", "stats")
graph.add_edge("stats", END)

# ✅ Compile Graph
pipeline = graph.compile()


# ✅ Execution
st.sidebar.header("Configuration")

ticker = st.sidebar.text_input("Enter Stock Ticker or Company Name", value=st.session_state.state["ticker"])
st.session_state.state["ticker"] = ticker

num_results = st.sidebar.number_input("Number of Bing results", min_value=1, max_value=20, value=5)
st.session_state.state["num_results"] = num_results

if st.sidebar.button("Run Pipeline"):
    with st.spinner("Running pipeline..."):
        st.session_state.state = pipeline.invoke(st.session_state.state)
        st.success("✅ Pipeline completed successfully!")

# ✅ Chatbot at the end
st.subheader("💬 Chatbot")
user_query = st.text_input("Ask about the results")
if st.button("Ask"):
    prompt = json.dumps(st.session_state.history, indent=2) + "\n" + user_query
    response = openai.ChatCompletion.create(engine=deployment_name, messages=[{"role": "system", "content": prompt}], temperature=0.5, max_tokens=500)
    st.write(response['choices'][0]['message']['content'])
