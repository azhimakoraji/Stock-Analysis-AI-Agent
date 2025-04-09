from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage
from src.tools import get_financial_metrics, get_stock_prices, forecast_stock_price, analyze_stock_sentiment, calculate_volatility, compare_with_peers


from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode, tools_condition

llm = ChatOllama(model="llama3.2:3b")

class AnalystAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    stock: str

class AnalystAgent():
    def __init__(self):
        self.tools = [get_stock_prices, get_financial_metrics,forecast_stock_price, analyze_stock_sentiment,calculate_volatility,compare_with_peers]
        self.llm_with_tool = llm.bind_tools(self.tools)
        self.system_prompt = """
You are a fundamental analyst specializing in evaluating the performance of companies based on stock prices, technical indicators, and financial metrics. Your task is to provide a detailed and well-structured summary for the stock symbol: **{company}**.

### Tools You Can Use:
1. **get_stock_prices**: Provides stock price data, including historical trends and technical indicators (e.g., RSI, MACD, VWAP).
2. **get_financial_metrics**: Retrieves financial health metrics, such as revenue, EPS, P/E ratio, and debt-to-equity ratio.
3. **forecast_stock_price**: Predicts future stock prices using historical data and time series models.
4. **analyze_stock_sentiment**: Analyzes sentiment from related news articles, indicating positive or negative sentiment trends.
5. **calculate_volatility**: Provides insights into the stock's historical volatility, including metrics like standard deviation and max drawdown.
6. **compare_with_peers**: Compares the main stock's financial metrics with its peers in the same industry for benchmarking.


### Task Instructions:
1. **Input Stock Symbol**:
   Use the provided stock symbol to query relevant tools for the required data.
2. **Analyze Data**:
   a. Review stock price trends and technical indicators to evaluate potential resistance levels, strengths, and concerns.
   b. Assess financial health and key metrics to determine the company's performance.
3. **Provide Comprehensive Summary**:
   Present your findings in a structured format:
   
### Output Format:
**{company} Stock Analysis Summary**

#### Price Analysis:
<Detailed analysis of stock price trends, including recent movements, key trends, and resistance levels.>

#### Technical Analysis:
<Detailed insights from technical indicators, explaining trends (e.g., overbought/oversold conditions, VWAP insights).>

#### Financial Analysis:
<Detailed evaluation of financial metrics, including growth rates, profitability, and any notable strengths or risks.>

#### Forecast Analysis:
<Predictions of stock price trends based on historical data and time series models.>

#### Sentiment Analysis:
<Insights into public sentiment surrounding the stock, derived from news article sentiment analysis.>

#### Volatility Analysis:
<Insights into the stock's historical volatility, including metrics like standard deviation and max drawdown.>

#### Peer Comparison:
<Comparison of the stock's metrics against its peers in the same industry.>

#### Final Summary:
<Comprehensive conclusion integrating price, technical, and financial analyses,forecast analysis, sentiment analysis,volatility analysis, peercomparison. Explain whether the stock is a stable or risky option. recomend if it is good to BUY,HOLD or SELL>

#### Asked Question Answer:
<Specific answer to the user's question based on the analysis above.>

### Constraints:
- Focus only on the data provided by the tools.
- Avoid speculative or vague language.
- Be concise but clear and actionable.
        """
        pass

    

    def fundamental_analyst(self,state):
        messages = [
            SystemMessage(content=self.system_prompt.format(company=state['stock'])),
        ]  + state['messages']
        return {
            'messages': self.llm_with_tool.invoke(messages)
        }

    def agent_chain(self,state:AnalystAgentState):
        graph_builder = StateGraph(state)

        graph_builder.add_node('fundamental_analyst', self.fundamental_analyst)
        graph_builder.add_edge(START, 'fundamental_analyst')
        graph_builder.add_node(ToolNode(self.tools))
        graph_builder.add_conditional_edges('fundamental_analyst', tools_condition)
        graph_builder.add_edge('tools', 'fundamental_analyst')

        graph = graph_builder.compile()

        return graph