"""
Agent 1: Information Gathering
Sub-agents: News Analyst, Social Media Analyst, Market Analyst, Fundamental Analyst
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import requests
import json


class NewsAnalystAgent:
    """Fetches and scores news sentiment from Yahoo Finance RSS feeds."""

    def get_news(self, ticker: str) -> List[Dict]:
        try:
            stock = yf.Ticker(ticker)
            news = stock.news or []
            scored = []
            for item in news[:10]:
                title = item.get("content", {}).get("title", "")
                sentiment = self._score_sentiment(title)
                scored.append({
                    "title": title,
                    "published": item.get("content", {}).get("pubDate", ""),
                    "sentiment": sentiment,
                    "source": item.get("content", {}).get("provider", {}).get("displayName", "Yahoo Finance"),
                })
            return scored
        except Exception:
            return []

    def _score_sentiment(self, text: str) -> float:
        """Naive keyword-based sentiment scorer (-1 to +1)."""
        positive = ["beat", "surge", "soar", "record", "growth", "profit", "upgrade",
                    "buy", "strong", "gain", "rally", "breakout", "bullish", "exceed"]
        negative = ["miss", "drop", "fall", "loss", "decline", "downgrade", "sell",
                    "weak", "risk", "cut", "bearish", "concern", "warn", "below"]
        text_lower = text.lower()
        pos = sum(1 for w in positive if w in text_lower)
        neg = sum(1 for w in negative if w in text_lower)
        total = pos + neg
        if total == 0:
            return 0.0
        return round((pos - neg) / total, 2)


class SocialMediaAnalystAgent:
    """Simulates social sentiment scoring (Reddit/Twitter proxy via news volume)."""

    def get_sentiment_score(self, ticker: str, news_items: List[Dict]) -> Dict:
        if not news_items:
            return {"score": 0.0, "volume": 0, "trend": "neutral"}
        scores = [n["sentiment"] for n in news_items]
        avg = np.mean(scores)
        volume = len(news_items)
        trend = "bullish" if avg > 0.2 else ("bearish" if avg < -0.2 else "neutral")
        return {
            "score": round(float(avg), 2),
            "volume": volume,
            "trend": trend,
        }


class MarketAnalystAgent:
    """Fetches price action, momentum, and technical signals."""

    def analyze(self, ticker: str) -> Dict:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="6mo")
            if hist.empty:
                return {}

            close = hist["Close"]
            volume = hist["Volume"]

            # Moving averages
            ma50 = float(close.rolling(50).mean().iloc[-1])
            ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else ma50
            current = float(close.iloc[-1])

            # RSI
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss
            rsi = float(100 - (100 / (1 + rs.iloc[-1])))

            # Volume trend
            avg_vol_30 = float(volume.tail(30).mean())
            recent_vol = float(volume.tail(5).mean())
            vol_ratio = round(recent_vol / avg_vol_30, 2) if avg_vol_30 > 0 else 1.0

            # 52-week range
            high_52 = float(close.tail(252).max())
            low_52 = float(close.tail(252).min())

            # Momentum scores
            ret_1m = round((current / float(close.iloc[-22]) - 1) * 100, 2) if len(close) >= 22 else 0
            ret_3m = round((current / float(close.iloc[-66]) - 1) * 100, 2) if len(close) >= 66 else 0
            ret_6m = round((current / float(close.iloc[0]) - 1) * 100, 2)

            signal = "BUY" if (current > ma50 > ma200 and rsi < 70) else \
                     "SELL" if (current < ma50 < ma200 and rsi > 30) else "HOLD"

            return {
                "current_price": round(current, 2),
                "ma50": round(ma50, 2),
                "ma200": round(ma200, 2),
                "rsi": round(rsi, 2),
                "volume_ratio": vol_ratio,
                "high_52w": round(high_52, 2),
                "low_52w": round(low_52, 2),
                "return_1m": ret_1m,
                "return_3m": ret_3m,
                "return_6m": ret_6m,
                "signal": signal,
            }
        except Exception as e:
            return {"error": str(e)}


class FundamentalAnalystAgent:
    """Pulls key fundamentals: P/E, revenue growth, debt/equity, dividends, moat."""

    def analyze(self, ticker: str) -> Dict:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}

            # Revenue growth (5yr)
            rev_growth = self._get_revenue_growth(stock)

            forward_pe = info.get("forwardPE") or info.get("trailingPE")
            debt_eq = info.get("debtToEquity", 0)
            div_yield = info.get("dividendYield", 0) or 0
            payout = info.get("payoutRatio", 0) or 0
            roe = info.get("returnOnEquity", 0) or 0
            profit_margin = info.get("profitMargins", 0) or 0
            current_ratio = info.get("currentRatio", 0) or 0
            beta = info.get("beta", 1.0) or 1.0
            market_cap = info.get("marketCap", 0) or 0
            sector = info.get("sector", "Unknown")
            industry = info.get("industry", "Unknown")
            analyst_target = info.get("targetMeanPrice") or info.get("targetMedianPrice")
            current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
            rec = info.get("recommendationKey", "hold").upper()

            # Dividend sustainability (0-100)
            div_score = self._dividend_sustainability(div_yield, payout, roe, profit_margin)

            # Moat rating (0-10)
            moat = self._moat_rating(profit_margin, roe, market_cap, info)

            return {
                "forward_pe": round(forward_pe, 2) if forward_pe else None,
                "revenue_growth_5yr": rev_growth,
                "debt_to_equity": round(debt_eq / 100, 2) if debt_eq else 0,
                "dividend_yield": round(div_yield * 100, 2),
                "payout_ratio": round(payout * 100, 2),
                "dividend_sustainability": div_score,
                "roe": round(roe * 100, 2),
                "profit_margin": round(profit_margin * 100, 2),
                "current_ratio": round(current_ratio, 2),
                "beta": round(beta, 2),
                "market_cap": market_cap,
                "sector": sector,
                "industry": industry,
                "analyst_target": round(analyst_target, 2) if analyst_target else None,
                "current_price": round(current_price, 2) if current_price else None,
                "analyst_recommendation": rec,
                "moat_rating": moat,
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_revenue_growth(self, stock) -> float:
        try:
            fin = stock.financials
            if fin is None or fin.empty:
                return 0.0
            rev_row = None
            for idx in fin.index:
                if "revenue" in str(idx).lower() or "total revenue" in str(idx).lower():
                    rev_row = idx
                    break
            if rev_row is None:
                return 0.0
            rev = fin.loc[rev_row].dropna()
            if len(rev) < 2:
                return 0.0
            oldest, newest = float(rev.iloc[-1]), float(rev.iloc[0])
            if oldest <= 0:
                return 0.0
            years = len(rev) - 1
            cagr = (newest / oldest) ** (1 / years) - 1
            return round(cagr * 100, 2)
        except Exception:
            return 0.0

    def _dividend_sustainability(self, yield_: float, payout: float, roe: float, margin: float) -> int:
        score = 50
        if yield_ > 0:
            score += 10
        if payout < 0.6:
            score += 20
        elif payout > 0.9:
            score -= 20
        if roe > 0.15:
            score += 10
        if margin > 0.10:
            score += 10
        return max(0, min(100, score))

    def _moat_rating(self, margin: float, roe: float, market_cap: int, info: dict) -> float:
        score = 0
        if margin > 0.20:
            score += 2
        elif margin > 0.10:
            score += 1
        if roe > 0.20:
            score += 2
        elif roe > 0.10:
            score += 1
        if market_cap > 100_000_000_000:
            score += 2
        elif market_cap > 10_000_000_000:
            score += 1
        # Brand / pricing power proxy: gross margins
        gm = info.get("grossMargins", 0) or 0
        if gm > 0.50:
            score += 2
        elif gm > 0.30:
            score += 1
        # R&D investment proxy
        if info.get("sector") in ["Technology", "Healthcare"]:
            score += 1
        return min(10.0, round(score, 1))


class InformationGatherer:
    """Orchestrates all Agent 1 sub-agents."""

    def __init__(self):
        self.news_agent = NewsAnalystAgent()
        self.social_agent = SocialMediaAnalystAgent()
        self.market_agent = MarketAnalystAgent()
        self.fundamental_agent = FundamentalAnalystAgent()

    def gather(self, ticker: str) -> Dict:
        news = self.news_agent.get_news(ticker)
        social = self.social_agent.get_sentiment_score(ticker, news)
        market = self.market_agent.analyze(ticker)
        fundamentals = self.fundamental_agent.analyze(ticker)

        return {
            "ticker": ticker,
            "timestamp": datetime.utcnow().isoformat(),
            "news": news,
            "social_sentiment": social,
            "market": market,
            "fundamentals": fundamentals,
        }
