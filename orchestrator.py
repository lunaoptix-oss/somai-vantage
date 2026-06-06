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


def _enrich_analyst(result: Dict) -> Dict:
    """
    When Yahoo Finance blocks analyst data (cloud IP restriction), derive
    recommendation and target from our own 5-agent analysis.
    """
    fund = result.get("fundamentals", {})
    research = result.get("research", {})
    pt = result.get("price_targets", {})
    cls = research.get("classification", "NEUTRAL")
    score = research.get("composite_score", 0.5)

    # Fill analyst_recommendation from AI composite if missing
    if not fund.get("analyst_recommendation") or fund.get("analyst_recommendation") == "HOLD":
        if score >= 0.65:   rec = "STRONG BUY"
        elif score >= 0.55: rec = "BUY"
        elif score <= 0.35: rec = "STRONG SELL"
        elif score <= 0.45: rec = "SELL"
        else:               rec = "HOLD"
        fund["analyst_recommendation"] = rec

    # Fill analyst_target with a meaningful estimate when not available from Yahoo Finance
    if not fund.get("analyst_target"):
        current = pt.get("current") or fund.get("current_price", 0)
        if current and current > 0:
            # Estimate based on AI composite score — mirrors typical analyst upside
            if score >= 0.65:   mult = 1.25   # STRONG BUY: ~25% upside
            elif score >= 0.55: mult = 1.15   # BUY:        ~15% upside
            elif score >= 0.45: mult = 1.05   # HOLD:       ~5% upside
            elif score >= 0.35: mult = 0.95   # SELL:       ~5% downside
            else:               mult = 0.85   # STRONG SELL:~15% downside
            # Blend with 52w high if available
            high52 = result.get("market", {}).get("high_52w")
            if high52 and high52 > current:
                target = round((current * mult * 0.6 + high52 * 0.4), 2)
            else:
                target = round(current * mult, 2)
            fund["analyst_target"] = target

    result["fundamentals"] = fund
    return result


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

    result = {
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
    return _enrich_analyst(result)


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
