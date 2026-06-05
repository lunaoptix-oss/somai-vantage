"""
Agent 2: Researcher — segregates information, classifies bullish/bearish, computes composite score.
"""
from typing import Dict, List, Tuple
import numpy as np


class ResearcherAgent:

    WEIGHTS = {
        "news_sentiment": 0.15,
        "social_sentiment": 0.10,
        "technical_signal": 0.20,
        "fundamental_quality": 0.30,
        "momentum": 0.15,
        "valuation": 0.10,
    }

    def research(self, gathered: Dict) -> Dict:
        ticker = gathered["ticker"]
        market = gathered.get("market", {})
        fund = gathered.get("fundamentals", {})
        social = gathered.get("social_sentiment", {})
        news = gathered.get("news", [])

        scores = {}

        # News sentiment score (0-1)
        news_sentiments = [n.get("sentiment", 0) for n in news]
        scores["news_sentiment"] = (np.mean(news_sentiments) + 1) / 2 if news_sentiments else 0.5

        # Social sentiment (0-1)
        soc_score = social.get("score", 0)
        scores["social_sentiment"] = (soc_score + 1) / 2

        # Technical signal (0-1)
        sig = market.get("signal", "HOLD")
        rsi = market.get("rsi", 50)
        vol_ratio = market.get("volume_ratio", 1)
        tech = 0.5
        if sig == "BUY":
            tech = 0.75
        elif sig == "SELL":
            tech = 0.25
        if rsi < 40:
            tech = min(1.0, tech + 0.1)
        elif rsi > 65:
            tech = max(0.0, tech - 0.1)
        if vol_ratio > 1.5:
            tech = min(1.0, tech + 0.05)
        scores["technical_signal"] = round(tech, 3)

        # Fundamental quality (0-1)
        pe = fund.get("forward_pe")
        rev_growth = fund.get("revenue_growth_5yr", 0)
        de = fund.get("debt_to_equity", 1)
        moat = fund.get("moat_rating", 5)
        roe = fund.get("roe", 0)
        margin = fund.get("profit_margin", 0)

        fund_score = 0.5
        if pe and 5 < pe < 25:
            fund_score += 0.1
        elif pe and pe > 40:
            fund_score -= 0.1
        if rev_growth > 10:
            fund_score += 0.1
        elif rev_growth < 0:
            fund_score -= 0.1
        if de < 0.5:
            fund_score += 0.05
        elif de > 2:
            fund_score -= 0.1
        fund_score += (moat / 10) * 0.1
        if roe > 15:
            fund_score += 0.05
        if margin > 10:
            fund_score += 0.05
        scores["fundamental_quality"] = round(max(0, min(1, fund_score)), 3)

        # Momentum (0-1)
        r1 = market.get("return_1m", 0)
        r3 = market.get("return_3m", 0)
        r6 = market.get("return_6m", 0)
        mom = 0.5 + (r1 * 0.4 + r3 * 0.35 + r6 * 0.25) / 200
        scores["momentum"] = round(max(0, min(1, mom)), 3)

        # Valuation (0-1)
        analyst_target = fund.get("analyst_target")
        current = fund.get("current_price") or market.get("current_price")
        val_score = 0.5
        if analyst_target and current and current > 0:
            upside = (analyst_target - current) / current
            val_score = 0.5 + upside * 0.5
        scores["valuation"] = round(max(0, min(1, val_score)), 3)

        # Composite score
        composite = sum(scores[k] * self.WEIGHTS[k] for k in self.WEIGHTS)
        composite = round(composite, 3)

        # Classification
        if composite >= 0.65:
            classification = "STRONG_BULL"
        elif composite >= 0.55:
            classification = "BULL"
        elif composite <= 0.35:
            classification = "STRONG_BEAR"
        elif composite <= 0.45:
            classification = "BEAR"
        else:
            classification = "NEUTRAL"

        # Price targets
        price_targets = self._compute_price_targets(market, fund)

        return {
            "ticker": ticker,
            "composite_score": composite,
            "classification": classification,
            "sub_scores": scores,
            "price_targets": price_targets,
            "analyst_recommendation": fund.get("analyst_recommendation", "HOLD"),
        }

    def _compute_price_targets(self, market: Dict, fund: Dict) -> Dict:
        current = fund.get("current_price") or market.get("current_price")
        if not current or current <= 0:
            return {}   # skip if no price data

        analyst_target = fund.get("analyst_target") or current
        beta = fund.get("beta", 1.0)
        high_52 = market.get("high_52w", current * 1.3)
        low_52 = market.get("low_52w", current * 0.7)

        # Bull target: analyst target blended with 52w range
        bull = round((analyst_target * 0.6 + high_52 * 0.4), 2)
        # Bear target: downside scenario
        bear = round(current * (1 - 0.15 * beta), 2)
        # Entry zone: 2-5% below current
        entry_low = round(current * 0.97, 2)
        entry_high = round(current * 1.01, 2)
        # Stop loss: based on volatility proxy (beta)
        stop_loss = round(current * (1 - 0.08 * beta), 2)

        return {
            "current": round(current, 2),
            "bull_target": bull,
            "bear_target": bear,
            "analyst_target": round(analyst_target, 2),
            "entry_zone_low": entry_low,
            "entry_zone_high": entry_high,
            "stop_loss": stop_loss,
            "upside_pct": round((bull - current) / current * 100, 1) if current else 0,
            "downside_pct": round((bear - current) / current * 100, 1) if current else 0,
        }
