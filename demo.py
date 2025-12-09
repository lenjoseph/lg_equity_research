import streamlit as st
import requests
import time
from concurrent.futures import ThreadPoolExecutor

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Equity Research Agent",
    page_icon=None,
    layout="wide",
)

# Custom CSS for styling
st.markdown(
    """
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    .sentiment-bullish { color: #00d26a; font-weight: bold; }
    .sentiment-bearish { color: #ff6b6b; font-weight: bold; }
    .sentiment-neutral { color: #ffd93d; font-weight: bold; }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .metrics-container {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(139, 92, 246, 0.3);
        margin: 1rem 0;
    }
    .metrics-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #a78bfa;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
    }
    .metric-box {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #e0e7ff;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }
    .agent-metrics-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 1rem;
    }
    .agent-metrics-table th {
        background: rgba(139, 92, 246, 0.2);
        padding: 0.75rem;
        text-align: left;
        font-weight: 600;
        color: #c4b5fd;
        border-bottom: 2px solid rgba(139, 92, 246, 0.3);
    }
    .agent-metrics-table td {
        padding: 0.75rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        color: #e2e8f0;
    }
    .agent-metrics-table tr:hover {
        background: rgba(255, 255, 255, 0.03);
    }
    .model-badge {
        background: rgba(59, 130, 246, 0.2);
        color: #93c5fd;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-family: monospace;
    }
    .cached-badge {
        background: rgba(34, 197, 94, 0.2);
        color: #86efac;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.title("AI Equity Research Agent")
st.markdown("*Multi-agent equity analysis powered by LangGraph*")

# Initialize session state for results
if "results" not in st.session_state:
    st.session_state.results = None

# Main area input form
st.header("Research Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    ticker = st.text_input(
        "Stock Ticker",
        value="AAPL",
        placeholder="e.g., AAPL, MSFT, GOOGL",
    ).upper()

with col2:
    trade_direction = st.selectbox(
        "Trade Direction",
        options=["long", "short"],
        index=0,
    )

with col3:
    trade_duration = st.selectbox(
        "Trade Duration",
        options=["day_trade", "swing_trade", "position_trade"],
        index=2,
        format_func=lambda x: x.replace("_", " ").title(),
    )

analyze_button = st.button("Analyze", type="primary")

st.divider()

# Main content
if analyze_button:
    if not ticker:
        st.error("Please enter a stock ticker")
    else:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        def make_api_request():
            """Make the API request in a separate thread."""
            return requests.post(
                f"{API_URL}/research-equity",
                json={
                    "ticker": ticker,
                    "trade_direction": trade_direction,
                    "trade_duration": trade_duration,
                },
                timeout=240,
            )

        # Countdown intervals in seconds
        countdown_messages = [
            (0, "~1 minute remaining"),
            (15, "~45 seconds remaining"),
            (30, "~30 seconds remaining"),
            (45, "~15 seconds remaining"),
        ]

        status_text.info(f"Analyzing {ticker}... ~1 minute remaining")

        try:
            # Run API request in background thread
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(make_api_request)
                start_time = time.time()

                # Update countdown while waiting for response
                while not future.done():
                    elapsed = time.time() - start_time

                    # Find appropriate countdown message
                    current_message = countdown_messages[0][1]
                    for threshold, message in countdown_messages:
                        if elapsed >= threshold:
                            current_message = message

                    # Update progress bar (cap at 90% until complete)
                    progress = min(int((elapsed / 60) * 90), 90)
                    progress_bar.progress(progress)
                    status_text.info(f"Analyzing {ticker}... {current_message}")

                    time.sleep(1)

                # Get the response
                response = future.result()

            progress_bar.progress(100)
            status_text.empty()

            if response.status_code == 200:
                data = response.json()
                st.session_state.results = data

            elif response.status_code == 400:
                st.error(f"Invalid ticker: {ticker}")
            else:
                st.error(f"API Error: {response.status_code}")

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to API. Make sure the server is running on localhost:8000"
            )
        except requests.exceptions.Timeout:
            st.error("Request timed out. The analysis is taking too long.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Display results if available
if st.session_state.results:
    data = st.session_state.results

    # Success message
    st.success(f"Analysis complete for {data.get('ticker', ticker)}")

    # Metrics Section
    metrics = data.get("metrics", {})
    if metrics:
        with st.expander("📊 Performance Metrics", expanded=False):
            # Summary metrics row
            col1, col2, col3, col4 = st.columns(4)

            total_latency = metrics.get("total_latency_ms", 0)
            total_tokens = metrics.get("total_tokens", {})

            with col1:
                latency_display = (
                    f"{total_latency / 1000:.1f}s"
                    if total_latency >= 1000
                    else f"{total_latency:.0f}ms"
                )
                st.metric("Total Latency", latency_display)

            with col2:
                st.metric("Total Tokens", f"{total_tokens.get('total', 0):,}")

            with col3:
                st.metric("Input Tokens", f"{total_tokens.get('input', 0):,}")

            with col4:
                st.metric("Output Tokens", f"{total_tokens.get('output', 0):,}")

            st.markdown("---")

            # Agent-level metrics table
            st.markdown("**Agent Breakdown**")

            agent_metrics = metrics.get("agents", {})
            if agent_metrics:
                # Build table data
                table_data = []
                for agent_name, agent_data in agent_metrics.items():
                    latency = agent_data.get("latency_ms", 0)
                    tokens = agent_data.get("tokens", {})
                    model = agent_data.get("model", "N/A")
                    cached = agent_data.get("cached", False)

                    table_data.append(
                        {
                            "Agent": agent_name.replace("_", " ").title(),
                            "Latency": (
                                f"{latency / 1000:.2f}s"
                                if latency >= 1000
                                else f"{latency:.0f}ms"
                            ),
                            "Input": f"{tokens.get('input', 0):,}",
                            "Output": f"{tokens.get('output', 0):,}",
                            "Total": f"{tokens.get('total', 0):,}",
                            "Model": model or "N/A",
                            "Cached": "✓" if cached else "—",
                        }
                    )

                # Sort by agent name for consistent ordering
                table_data.sort(key=lambda x: x["Agent"])

                # Display as styled table using HTML
                table_html = """
                <table class="agent-metrics-table">
                    <thead>
                        <tr>
                            <th>Agent</th>
                            <th>Latency</th>
                            <th>Input Tokens</th>
                            <th>Output Tokens</th>
                            <th>Total Tokens</th>
                            <th>Model</th>
                            <th>Cached</th>
                        </tr>
                    </thead>
                    <tbody>
                """

                for row in table_data:
                    cached_class = "cached-badge" if row["Cached"] == "✓" else ""
                    table_html += f"""
                        <tr>
                            <td><strong>{row['Agent']}</strong></td>
                            <td>{row['Latency']}</td>
                            <td>{row['Input']}</td>
                            <td>{row['Output']}</td>
                            <td>{row['Total']}</td>
                            <td><span class="model-badge">{row['Model']}</span></td>
                            <td><span class="{cached_class}">{row['Cached']}</span></td>
                        </tr>
                    """

                table_html += """
                    </tbody>
                </table>
                """

                st.markdown(table_html, unsafe_allow_html=True)

    st.divider()

    # Combined Sentiment (main result)
    st.header("Investment Thesis")
    st.markdown(data.get("combined_sentiment", "No sentiment available"))

    st.divider()

    # Individual Agent Results
    st.header("Detailed Analysis")

    sentiment_data = data.get("sentiment_analysis", {})

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("Fundamental Analysis", expanded=True):
            st.markdown(sentiment_data.get("fundamental", "Not available"))

        with st.expander("Technical Analysis", expanded=True):
            st.markdown(sentiment_data.get("technical", "Not available"))

        with st.expander("Macro Analysis", expanded=True):
            st.markdown(sentiment_data.get("macro", "Not available"))

    with col2:
        with st.expander("Industry Analysis", expanded=True):
            st.markdown(sentiment_data.get("industry", "Not available"))

        with st.expander("Peer Analysis", expanded=True):
            st.markdown(sentiment_data.get("peer", "Not available"))

        with st.expander("Headline Analysis", expanded=True):
            st.markdown(sentiment_data.get("headline", "Not available"))

        with st.expander("SEC Filings Analysis", expanded=True):
            st.markdown(sentiment_data.get("filings", "Not available"))
