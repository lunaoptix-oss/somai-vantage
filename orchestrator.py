"""
Orchestrator — runs all 5 agents for a given ticker and returns a unified result.
"""
import asyncio
from typing import Dict, List
from agents.agent1_gatherer import InformationGatherer
from agents.agent2_researcher import ResearcherAgent
from agents.agent3_risk import RiskManagementAgent
from agents.agent4_execution import ExecutionAgent
from agents.agent5_feedback import FeedbackAgent


gatherer = InformationGatherer()
researcher = ResearcherAgent()
risk_mgr = RiskManagementAgent()
executor = ExecutionAgent()
feedback = FeedbackAgent()


def analyze_ticker(ticker: str) -> Dict:
    ticker = ticker.upper().strip()

    # Agent 1: Gather
    gathered = gatherer.gather(ticker)

    # Agent 2: Research
    research = researcher.research(gathered)

    # Agent 3: Risk
    risk = risk_mgr.evaluate(gathered, research)

    # Agent 4: Execution signal
    execution = executor.generate_signal(gathered, research, risk)

    # Agent 5: Record signal
    signal_id = feedback.record_signal(execution, research)

    # Agent 1 refinements from feedback loop
    refinements = feedback.get_agent1_refinements()

    return {
        "ticker": ticker,
        "signal_id": signal_id,
        "summary": {
            "action": execution["action"],
            "confidence": execution["confidence"],
            "confidence_label": execution["confidence_label"],
            "classification": research["classification"],
            "composite_score": research["composite_score"],
            "risk_score": risk["risk_score"],
            "risk_label": risk["risk_label"],
            "timeframe": execution["timeframe"],
        },
        "fundamentals": gathered["fundamentals"],
        "market": gathered["market"],
        "price_targets": research["price_targets"],
        "execution": execution,
        "risk": risk,
        "research": research,
        "news": gathered["news"][:5],
        "social_sentiment": gathered["social_sentiment"],
        "refinements": refinements,
    }


async def analyze_tickers_async(tickers: List[str]) -> List[Dict]:
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(None, analyze_ticker, t) for t in tickers]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    out = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            out.append({"ticker": tickers[i], "error": str(r)})
        else:
            out.append(r)
    return out


# Default universe of stocks to screen
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "AVGO", "JPM", "V",
    "UNH", "LLY", "XOM", "MA", "HD",
    "ORCL", "COST", "AMD", "CRM", "NFLX",
]
