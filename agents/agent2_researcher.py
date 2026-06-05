"""
Agent 2: Researcher — classifies stocks bullish/bearish, computes composite score.
All arithmetic uses safe n() helper so None values never cause crashes.
"""
from typing import Dict
import numpy as np


def n(val, default=0.0):
    """Return float(val) or default if val is None/NaN."""
    try:
        v = float(val)
        return default if (v != v) else v   # NaN check
    except (TypeError, ValueError):
        return default


class ResearcherAgent:

    WEIGHTS = {
        "news_sentiment":      0.15,
        "social_sentiment":    0.10,
        "technical_signal":    0.20,
        "fundamental_quality": 0.30,
        "momentum":            0.15,
        "valuation":           0.10,
    }

    def research(self, gathered: Dict) -> Dict:
        ticker  = gathered["ticker"]
        market  = gathered.get("market", {})
        fund    = gathered.get("fundamentals", {})
        social  = gathered.get("social_sentiment", {})
        news    = gathered.get("news", [])

        # Require a valid price
        current = n(fund.get("current_price")) or n(market.get("current_price"))
        if current <= 0:
            return {
                "ticker": ticker, "composite_score": 0.5,
                "classification": "NEUTRAL", "sub_scores": {},
                "price_targets": {}, "analyst_recommendation": "HOLD",
            }

        scores = {}

        # News sentiment (0-1)
        news_sents = [n(x.get("sentiment")) for x in news]
        scores["news_sentiment"] = (np.mean(news_sents) + 1) / 2 if news_sents else 0.5

        # Social sentiment (0-1)
        scores["social_sentiment"] = (n(social.get("score")) + 1) / 2

        # Technical signal (0-1)
        sig       = market.get("signal", "HOLD")
        rsi       = n(market.get("rsi"), 50)
        vol_ratio = n(market.get("volume_ratio"), 1)
        tech = 0.75 if sig == "BUY" else 0.25 if sig == "SELL" else 0.50
        if rsi < 40:  tech = min(1.0, tech + 0.10)
        if rsi > 65:  tech = max(0.0, tech - 0.10)
        if vol_ratio > 1.5: tech = min(1.0, tech + 0.05)
        scores["technical_signal"] = round(tech, 3)

        # Fundamental quality (0-1)
        pe     = n(fund.get("forward_pe"))
        rev_g  = n(fund.get("revenue_growth_5yr"))
        de     = n(fund.get("debt_to_equity"), 1)
        moat   = n(fund.get("moat_rating"), 5)
        roe    = n(fund.get("roe"))
        margin = n(fund.get("profit_margin"))

        fq = 0.50
        if 0 < pe < 25:   fq += 0.10
        elif pe > 40:      fq -= 0.10
        if rev_g > 10:     fq += 0.10
        elif rev_g < 0:    fq -= 0.10
        if de < 0.5:       fq += 0.05
        elif de > 2:       fq -= 0.10
        fq += (moat / 10) * 0.10
        if roe    > 15:    fq += 0.05
        if margin > 10:    fq += 0.05
        scores["fundamental_quality"] = round(max(0, min(1, fq)), 3)

        # Momentum (0-1)
        r1 = n(market.get("return_1m"))
        r3 = n(market.get("return_3m"))
        r6 = n(market.get("return_6m"))
        mom = 0.5 + (r1 * 0.40 + r3 * 0.35 + r6 * 0.25) / 200
        scores["momentum"] = round(max(0, min(1, mom)), 3)

        # Valuation (0-1)
        analyst_tgt = n(fund.get("analyst_target"))
        val = 0.5
        if analyst_tgt > 0 and current > 0:
            upside = (analyst_tgt - current) / current
            val = max(0, min(1, 0.5 + upside * 0.5))
        scores["valuation"] = round(val, 3)

        # Composite
        composite = round(sum(scores[k] * self.WEIGHTS[k] for k in self.WEIGHTS), 3)

        if composite >= 0.65:   cls = "STRONG_BULL"
        elif composite >= 0.55: cls = "BULL"
        elif composite <= 0.35: cls = "STRONG_BEAR"
        elif composite <= 0.45: cls = "BEAR"
        else:                   cls = "NEUTRAL"

        return {
            "ticker":               ticker,
            "composite_score":      composite,
            "classification":       cls,
            "sub_scores":           scores,
            "price_targets":        self._price_targets(current, market, fund),
            "analyst_recommendation": fund.get("analyst_recommendation", "HOLD"),
        }

    def _price_targets(self, current: float, market: Dict, fund: Dict) -> Dict:
        analyst_tgt = n(fund.get("analyst_target")) or current
        beta        = n(fund.get("beta"), 1.0)
        high_52     = n(market.get("high_52w")) or current * 1.3
        low_52      = n(market.get("low_52w"))  or current * 0.7

        bull        = round(analyst_tgt * 0.6 + high_52 * 0.4, 2)
        bear        = round(current * (1 - 0.15 * beta), 2)
        entry_low   = round(current * 0.97, 2)
        entry_high  = round(current * 1.01, 2)
        stop_loss   = round(current * (1 - 0.08 * beta), 2)
        upside_pct  = round((bull - current) / current * 100, 1) if current else 0
        down_pct    = round((bear - current) / current * 100, 1) if current else 0

        return {
            "current":        round(current, 2),
            "bull_target":    bull,
            "bear_target":    bear,
            "analyst_target": round(analyst_tgt, 2),
            "entry_zone_low": entry_low,
            "entry_zone_high":entry_high,
            "stop_loss":      stop_loss,
            "upside_pct":     upside_pct,
            "downside_pct":   down_pct,
        }
