
# 📊 Stock Analysis AI Agent

An interactive stock analysis tool powered by LLMs and Streamlit. It enables financial analysts, investors, and decision-makers to generate AI-based stock market insights by simply querying the system with a stock symbol and a question.

---

## 🧠 Features

- 📈 **Historical Stock Analysis**: Calculates indicators like RSI, MACD, VWAP
- 🧮 **Financial Metrics**: Extracts P/E, profit margins, price-to-book ratios
- 💬 **News Sentiment Analysis**: Pulls news headlines and performs sentiment detection
- 🔮 **Forecasting**: Predicts stock prices using Exponential Smoothing
- ⚠️ **Volatility Metrics**: Measures annualized volatility and drawdown
- 📊 **Peer Comparison**: Benchmarks target stock against similar companies

---

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain & LangGraph
- Financial APIs (Yahoo Finance, News APIs)
- Exponential Smoothing, Pandas, NumPy

---

## 📂 Project Structure

```
final_project_LLM/
│
├── main.py                     # Streamlit app entry point
├── workflow.py                 # Agent flow orchestration
├── Stock Analysis AI Agent.pptx # Presentation/demo slide
├── example.png                 # Demo screenshot or diagram
│
├── src/                        # Core logic and tools
│   ├── agent.py                # LLM Agent and decision flow
│   └── tools.py                # Stock analysis tools and helpers
│
├── Financial Analyst/          # Diagram files or references
├── __pycache__/                # Cache files
└── .ipynb_checkpoints/         # Jupyter auto-save files
```

---

## 🚀 Getting Started

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/stock-analysis-ai-agent.git
cd stock-analysis-ai-agent
```

2. Install required libraries:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run main.py
```

Make sure to configure your API keys if needed in an `.env` file.

---

## 📸 Demo

Refer to `Stock Analysis AI Agent.pptx` and `example.png` in the project for a full walkthrough.

---

## 🙋‍♂️ Author

**Azhima Koraji**  
Built as part of the Qarir Academy Final Project on LLMs.

---

