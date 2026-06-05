"""
Agent 4: Execution — trade signal generation. Fully None-safe.
"""
from typing import Dict, List
from datetime import datetime


def n(val, default=0.0):
    try:
        v = float(val)
        return default if (v != v) else v
    except (TypeError, ValueError):
        return default


class ExecutionAgent:

    def generate_signal(self, gathered: Dict, research: Dict, risk: Dict) -> Dict:
        ticker  = gathered["ticker"]
        cls     = research.get("classification", "NEUTRAL")
        rs      = n(risk.get("risk_score"), 5)
        rr      = n(risk.get("risk_reward_ratio"), 1.0)
        pt      = research.get("price_targets", {})
        fund    = gathered.get("fundamentals", {})
        market  = gathered.get("market", {})

        action     = self._action(cls, rs, rr)
        confidence = self._confidence(research, risk)

        return {
            "ticker":           ticker,
            "action":           action,
            "confidence":       confidence,
            "confidence_label": "HIGH" if confidence >= 80 else "MEDIUM" if confidence >= 60 else "LOW",
            "timeframe":        self._timeframe(market, fund),
            "entry_price_low":  pt.get("entry_zone_low"),
            "entry_price_high": pt.get("entry_zone_high"),
            "target_price":     pt.get("bull_target"),
            "stop_loss":        pt.get("stop_loss"),
            "position_size_pct":n(risk.get("suggested_position_size_pct"), 2),
            "risk_reward":      rr,
            "rationale":        self._rationale(cls, research, risk, fund, market),
            "generated_at":     datetime.utcnow().isoformat(),
        }

    def _action(self, cls, rs, rr):
        if cls == "STRONG_BULL" and rs <= 7 and rr >= 1.5: return "STRONG BUY"
        if cls in ("BULL", "STRONG_BULL") and rs <= 8:      return "BUY"
        if cls == "STRONG_BEAR" and rs <= 7:                return "STRONG SELL"
        if cls in ("BEAR", "STRONG_BEAR"):                  return "SELL"
        return "HOLD / WATCH"

    def _confidence(self, research, risk):
        comp        = n(research.get("composite_score"), 0.5)
        risk_score  = n(risk.get("risk_score"), 5)
        risk_factor = 1 - (risk_score - 1) / 18
        return round(min(1.0, max(0.0, comp * 0.7 + risk_factor * 0.3)) * 100, 1)

    def _timeframe(self, market, fund):
        beta = n(fund.get("beta"), 1.0)
        r1m  = n(market.get("return_1m"))
        if beta > 1.3 and abs(r1m) > 10:
            return "SHORT-TERM (1-4 weeks)"
        div_yield = n(fund.get("dividend_yield"))
        moat      = n(fund.get("moat_rating"))
        if div_yield > 2 or moat > 7:
            return "LONG-TERM (6-18 months)"
        return "MEDIUM-TERM (1-6 months)"

    def _rationale(self, cls, research, risk, fund, market) -> List[str]:
        sub    = research.get("sub_scores", {})
        points = []

        if n(sub.get("technical_signal"), 0.5) > 0.65:
            points.append(f"Technical setup: {market.get('signal','HOLD')} — strong price momentum")
        if n(sub.get("fundamental_quality"), 0.5) > 0.65:
            moat = n(fund.get("moat_rating"))
            roe  = n(fund.get("roe"))
            points.append(f"Strong fundamentals (moat: {moat:.0f}/10, ROE: {roe:.1f}%)")
        if n(sub.get("momentum"), 0.5) > 0.6:
            r3 = n(market.get("return_3m"))
            points.append(f"Positive price momentum ({r3:+.1f}% over 3 months)")
        if n(sub.get("valuation"), 0.5) > 0.6:
            upside = n(research.get("price_targets", {}).get("upside_pct"))
            points.append(f"Analyst consensus implies {upside:+.1f}% upside")
        if not points:
            points.append(f"Mixed signals — classified as {cls.replace('_',' ')}")

        rf = risk.get("risk_factors", [])
        if rf:
            points.append(f"Key risk: {rf[0]}")
        return points
