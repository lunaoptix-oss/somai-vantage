"""
Agent 3: Risk Management — 1-10 risk score, position sizing, stop-loss.
All arithmetic None-safe via n() helper.
"""
from typing import Dict


def n(val, default=0.0):
    try:
        v = float(val)
        return default if (v != v) else v
    except (TypeError, ValueError):
        return default


class RiskManagementAgent:

    def evaluate(self, gathered: Dict, research: Dict) -> Dict:
        market  = gathered.get("market", {})
        fund    = gathered.get("fundamentals", {})
        pt      = research.get("price_targets", {})
        cls     = research.get("classification", "NEUTRAL")

        risk_score, factors = self._score(market, fund, pt, cls)

        win_prob = n(research.get("composite_score"), 0.5)
        avg_win  = abs(n(pt.get("upside_pct"),   10)) / 100
        avg_loss = abs(n(pt.get("downside_pct"),  8)) / 100
        kelly    = self._kelly(win_prob, avg_win, avg_loss)

        beta     = n(fund.get("beta"), 1.0)
        return {
            "ticker":                    gathered["ticker"],
            "risk_score":                risk_score,
            "risk_factors":              factors,
            "risk_label":                self._label(risk_score),
            "suggested_position_size_pct": round(kelly * 100, 1),
            "max_drawdown_estimate_pct": round(beta * 25, 1),
            "stop_loss":                 pt.get("stop_loss"),
            "entry_zone": {
                "low":  pt.get("entry_zone_low"),
                "high": pt.get("entry_zone_high"),
            },
            "risk_reward_ratio": self._rr(pt),
        }

    def _score(self, market, fund, pt, cls):
        score   = 5.0
        factors = []

        beta = n(fund.get("beta"), 1.0)
        if beta > 1.5:
            score += 1.5
            factors.append(f"High beta ({beta:.2f}) - amplified volatility")
        elif beta < 0.7:
            score -= 0.5
            factors.append(f"Low beta ({beta:.2f}) - defensive stock")

        de = n(fund.get("debt_to_equity"))
        if de > 2:
            score += 1.5
            factors.append(f"High leverage (D/E: {de:.2f}x)")
        elif de < 0.3:
            score -= 0.5
            factors.append(f"Low debt (D/E: {de:.2f}x)")

        rsi = n(market.get("rsi"), 50)
        if rsi > 75:
            score += 1.0
            factors.append(f"Overbought RSI ({rsi:.0f})")
        elif rsi < 30:
            score += 0.5
            factors.append(f"Oversold RSI ({rsi:.0f}) - reversal risk")

        current  = n(pt.get("current"))
        high_52  = n(market.get("high_52w")) or current
        if current > 0 and high_52 > 0:
            pct_from_high = (high_52 - current) / high_52 * 100
            if pct_from_high < 5:
                score += 0.5
                factors.append("Trading near 52-week high")

        if cls in ("STRONG_BULL", "BULL"):   score -= 0.5
        elif cls in ("STRONG_BEAR", "BEAR"): score += 0.5

        pe = n(fund.get("forward_pe"))
        if pe > 40:
            score += 0.5
            factors.append(f"Rich valuation (P/E: {pe:.1f}x)")

        return round(max(1.0, min(10.0, score)), 1), factors

    def _label(self, s):
        if s <= 3: return "LOW"
        if s <= 5: return "MODERATE"
        if s <= 7: return "HIGH"
        return "VERY HIGH"

    def _kelly(self, win_p, avg_win, avg_loss):
        if avg_win <= 0 or avg_loss <= 0:
            return 0.02
        kelly = (win_p * avg_win - (1 - win_p) * avg_loss) / avg_win
        return round(max(0.01, min(0.10, kelly / 2)), 3)

    def _rr(self, pt):
        up   = abs(n(pt.get("upside_pct")))
        down = abs(n(pt.get("downside_pct"), 1))
        return round(up / down, 2) if down > 0 else 0.0
