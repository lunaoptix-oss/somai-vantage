"""
Agent 3: Risk Management — evaluates trade risk and assigns a 1-10 risk score.
"""
from typing import Dict
import math


class RiskManagementAgent:

    def evaluate(self, gathered: Dict, research: Dict) -> Dict:
        market = gathered.get("market", {})
        fund = gathered.get("fundamentals", {})
        pt = research.get("price_targets", {})
        classification = research.get("classification", "NEUTRAL")

        risk_score, factors = self._compute_risk(market, fund, pt, classification)

        # Position sizing (Kelly-lite: fraction of portfolio)
        win_prob = research.get("composite_score", 0.5)
        avg_win = abs(pt.get("upside_pct", 10)) / 100
        avg_loss = abs(pt.get("downside_pct", 8)) / 100
        kelly = self._kelly_fraction(win_prob, avg_win, avg_loss)

        # Max drawdown estimate
        beta = fund.get("beta", 1.0)
        max_drawdown = round(beta * 25, 1)  # proxy: beta * market's ~25% worst drawdown

        return {
            "ticker": gathered["ticker"],
            "risk_score": risk_score,
            "risk_factors": factors,
            "risk_label": self._risk_label(risk_score),
            "suggested_position_size_pct": round(kelly * 100, 1),
            "max_drawdown_estimate_pct": max_drawdown,
            "stop_loss": pt.get("stop_loss"),
            "entry_zone": {
                "low": pt.get("entry_zone_low"),
                "high": pt.get("entry_zone_high"),
            },
            "risk_reward_ratio": self._rr_ratio(pt),
        }

    def _compute_risk(self, market: Dict, fund: Dict, pt: Dict, classification: str):
        score = 5.0
        factors = []

        beta = fund.get("beta", 1.0)
        if beta > 1.5:
            score += 1.5
            factors.append(f"High beta ({beta:.2f}) — amplified market moves")
        elif beta < 0.7:
            score -= 0.5
            factors.append(f"Low beta ({beta:.2f}) — defensive stock")

        de = fund.get("debt_to_equity", 0)
        if de > 2:
            score += 1.5
            factors.append(f"High leverage (D/E: {de:.2f})")
        elif de < 0.3:
            score -= 0.5
            factors.append(f"Low debt (D/E: {de:.2f}) — strong balance sheet")

        rsi = market.get("rsi", 50)
        if rsi > 75:
            score += 1.0
            factors.append(f"Overbought RSI ({rsi:.0f})")
        elif rsi < 30:
            score += 0.5
            factors.append(f"Oversold RSI ({rsi:.0f}) — potential reversal")

        current = pt.get("current", 0)
        high_52 = market.get("high_52w", current)
        if current and high_52 and current > 0:
            pct_from_high = (high_52 - current) / high_52 * 100
            if pct_from_high < 5:
                score += 0.5
                factors.append("Trading near 52-week high — limited upside buffer")

        if classification in ("STRONG_BULL", "BULL"):
            score -= 0.5
        elif classification in ("STRONG_BEAR", "BEAR"):
            score += 0.5

        pe = fund.get("forward_pe")
        if pe and pe > 40:
            score += 0.5
            factors.append(f"Rich valuation (P/E: {pe:.1f}) — priced for perfection")

        score = round(max(1.0, min(10.0, score)), 1)
        return score, factors

    def _risk_label(self, score: float) -> str:
        if score <= 3:
            return "LOW"
        elif score <= 5:
            return "MODERATE"
        elif score <= 7:
            return "HIGH"
        return "VERY HIGH"

    def _kelly_fraction(self, win_prob: float, avg_win: float, avg_loss: float) -> float:
        if avg_loss <= 0:
            return 0.02
        kelly = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
        # Use half-Kelly for safety, cap at 10%
        half_kelly = kelly / 2
        return round(max(0.01, min(0.10, half_kelly)), 3)

    def _rr_ratio(self, pt: Dict) -> float:
        upside = abs(pt.get("upside_pct", 0))
        downside = abs(pt.get("downside_pct", 1))
        if downside == 0:
            return 0
        return round(upside / downside, 2)
