"""
Agent 4: Execution — generates actionable trade signals with precise entry/exit levels.
"""
from typing import Dict, List
from datetime import datetime


class ExecutionAgent:

    def generate_signal(self, gathered: Dict, research: Dict, risk: Dict) -> Dict:
        ticker = gathered["ticker"]
        classification = research.get("classification", "NEUTRAL")
        risk_score = risk.get("risk_score", 5)
        rr = risk.get("risk_reward_ratio", 1.0)
        pt = research.get("price_targets", {})
        fund = gathered.get("fundamentals", {})
        market = gathered.get("market", {})

        # Determine action
        action = self._determine_action(classification, risk_score, rr)

        # Confidence
        confidence = self._confidence(research, risk)

        # Timeframe suggestion
        timeframe = self._suggest_timeframe(market, fund)

        return {
            "ticker": ticker,
            "action": action,
            "confidence": confidence,
            "confidence_label": self._confidence_label(confidence),
            "timeframe": timeframe,
            "entry_price_low": pt.get("entry_zone_low"),
            "entry_price_high": pt.get("entry_zone_high"),
            "target_price": pt.get("bull_target"),
            "stop_loss": pt.get("stop_loss"),
            "position_size_pct": risk.get("suggested_position_size_pct"),
            "risk_reward": rr,
            "rationale": self._build_rationale(classification, research, risk, fund, market),
            "generated_at": datetime.utcnow().isoformat(),
        }

    def _determine_action(self, classification: str, risk_score: float, rr: float) -> str:
        if classification == "STRONG_BULL" and risk_score <= 7 and rr >= 1.5:
            return "STRONG BUY"
        elif classification in ("BULL", "STRONG_BULL") and risk_score <= 8:
            return "BUY"
        elif classification == "STRONG_BEAR" and risk_score <= 7:
            return "STRONG SELL"
        elif classification in ("BEAR", "STRONG_BEAR"):
            return "SELL"
        return "HOLD / WATCH"

    def _confidence(self, research: Dict, risk: Dict) -> float:
        comp = research.get("composite_score", 0.5)
        risk_factor = 1 - (risk.get("risk_score", 5) - 1) / 18
        confidence = (comp * 0.7 + risk_factor * 0.3)
        return round(min(1.0, max(0.0, confidence)) * 100, 1)

    def _confidence_label(self, confidence: float) -> str:
        if confidence >= 80:
            return "HIGH"
        elif confidence >= 60:
            return "MEDIUM"
        return "LOW"

    def _suggest_timeframe(self, market: Dict, fund: Dict) -> str:
        beta = fund.get("beta", 1.0)
        r1m = market.get("return_1m", 0)
        if beta > 1.3 and abs(r1m) > 10:
            return "SHORT-TERM (1-4 weeks)"
        elif fund.get("dividend_yield", 0) > 2 or fund.get("moat_rating", 0) > 7:
            return "LONG-TERM (6-18 months)"
        return "MEDIUM-TERM (1-6 months)"

    def _build_rationale(self, classification: str, research: Dict, risk: Dict,
                         fund: Dict, market: Dict) -> List[str]:
        points = []
        sub = research.get("sub_scores", {})

        if sub.get("technical_signal", 0.5) > 0.65:
            sig = market.get("signal", "HOLD")
            points.append(f"Technical setup: {sig} — price above MA50/MA200")
        if sub.get("fundamental_quality", 0.5) > 0.65:
            moat = fund.get("moat_rating", 0)
            points.append(f"Strong fundamentals (moat: {moat}/10, ROE: {fund.get('roe', 0):.1f}%)")
        if sub.get("momentum", 0.5) > 0.6:
            r3 = market.get("return_3m", 0)
            points.append(f"Positive price momentum (+{r3:.1f}% over 3 months)")
        if sub.get("valuation", 0.5) > 0.6:
            upside = research.get("price_targets", {}).get("upside_pct", 0)
            points.append(f"Analyst consensus implies {upside:.1f}% upside")
        if not points:
            points.append(f"Mixed signals — classified as {classification}")

        risk_factors = risk.get("risk_factors", [])
        if risk_factors:
            points.append(f"Key risk: {risk_factors[0]}")
        return points
