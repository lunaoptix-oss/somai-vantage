"""
Agent 1: Information Gathering
Sub-agents: News Analyst, Social Media Analyst, Market Analyst, Fundamental Analyst
"""
import yfinance as yf
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List

# ── Browser session so Yahoo Finance doesn't block cloud server IPs ──────────
def _make_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    })
    return s

def get_ticker(symbol: str):
    return yf.Ticker(symbol, session=_make_session())


# ── News Analyst ─────────────────────────────────────────────────────────────
class NewsAnalystAgent:
    POSITIVE = ["beat","surge","soar","record","growth","profit","upgrade","buy",
                "strong","gain","rally","breakout","bullish","exceed","raised","outperform"]
    NEGATIVE = ["miss","drop","fall","loss","decline","downgrade","sell","weak",
                "risk","cut","bearish","concern","warn","below","disappoints","layoff"]

    def get_news(self, ticker: str) -> List[Dict]:
        try:
            stock = get_ticker(ticker)
            news  = stock.news or []
            out   = []
            for item in news[:10]:
                c     = item.get("content", {}) if isinstance(item.get("content"), dict) else item
                title = c.get("title", item.get("title", ""))
                pub   = c.get("pubDate", item.get("providerPublishTime", ""))
                src   = (c.get("provider", {}) or {}).get("displayName", "Yahoo Finance")
                if title:
                    out.append({"title": title, "published": str(pub),
                                "sentiment": self._score(title), "source": src})
            return out
        except Exception:
            return []

    def _score(self, text: str) -> float:
        t   = text.lower()
        pos = sum(1 for w in self.POSITIVE if w in t)
        neg = sum(1 for w in self.NEGATIVE if w in t)
        tot = pos + neg
        return round((pos - neg) / tot, 2) if tot else 0.0


# ── Social Media Analyst ─────────────────────────────────────────────────────
class SocialMediaAnalystAgent:
    def get_sentiment_score(self, ticker: str, news: List[Dict]) -> Dict:
        if not news:
            return {"score": 0.0, "volume": 0, "trend": "neutral"}
        scores = [n["sentiment"] for n in news]
        avg    = float(np.mean(scores))
        trend  = "bullish" if avg > 0.15 else ("bearish" if avg < -0.15 else "neutral")
        return {"score": round(avg, 2), "volume": len(news), "trend": trend}


# ── Market Analyst ───────────────────────────────────────────────────────────
class MarketAnalystAgent:
    def analyze(self, ticker: str) -> Dict:
        try:
            stock = get_ticker(ticker)
            hist  = stock.history(period="6mo")
            if hist.empty:
                return {}
            close  = hist["Close"]
            volume = hist["Volume"]
            current = float(close.iloc[-1])

            ma50  = float(close.rolling(50).mean().iloc[-1])
            ma200 = float(close.rolling(min(200, len(close))).mean().iloc[-1])

            delta = close.diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rs    = gain / loss.replace(0, np.nan)
            rsi   = float(100 - (100 / (1 + rs.iloc[-1])))

            avg_vol  = float(volume.tail(30).mean())
            vol_ratio = round(float(volume.tail(5).mean()) / avg_vol, 2) if avg_vol > 0 else 1.0

            high_52 = float(close.tail(252).max())
            low_52  = float(close.tail(252).min())

            def ret(n):
                return round((current / float(close.iloc[-n]) - 1) * 100, 2) if len(close) >= n else 0.0

            signal = ("BUY"  if current > ma50 > ma200 and rsi < 72 else
                      "SELL" if current < ma50 < ma200 and rsi > 28 else "HOLD")

            return {
                "current_price": round(current, 2),
                "ma50":          round(ma50,    2),
                "ma200":         round(ma200,   2),
                "rsi":           round(rsi,     2),
                "volume_ratio":  vol_ratio,
                "high_52w":      round(high_52, 2),
                "low_52w":       round(low_52,  2),
                "return_1m":     ret(22),
                "return_3m":     ret(66),
                "return_6m":     ret(len(close) - 1),
                "signal":        signal,
            }
        except Exception as e:
            return {"error": str(e)}


# ── Fundamental Analyst ──────────────────────────────────────────────────────
class FundamentalAnalystAgent:
    def analyze(self, ticker: str) -> Dict:
        try:
            stock = get_ticker(ticker)
            info  = stock.info or {}

            price = (info.get("currentPrice")
                     or info.get("regularMarketPrice")
                     or info.get("previousClose")
                     or 0)

            forward_pe   = info.get("forwardPE") or info.get("trailingPE")
            debt_eq      = (info.get("debtToEquity") or 0) / 100
            div_yield    = info.get("dividendYield")   or 0
            payout       = info.get("payoutRatio")     or 0
            roe          = info.get("returnOnEquity")  or 0
            margin       = info.get("profitMargins")   or 0
            curr_ratio   = info.get("currentRatio")    or 0
            beta         = info.get("beta")            or 1.0
            market_cap   = info.get("marketCap")       or 0
            sector       = info.get("sector",   "Unknown")
            industry     = info.get("industry", "Unknown")
            analyst_tgt  = info.get("targetMeanPrice") or info.get("targetMedianPrice")
            analyst_rec  = (info.get("recommendationKey") or "hold").upper().replace("_", " ")
            gross_margin = info.get("grossMargins") or 0

            rev_growth   = self._revenue_cagr(stock)
            div_score    = self._div_sustainability(div_yield, payout, roe, margin)
            moat         = self._moat(margin, roe, market_cap, gross_margin, sector)

            return {
                "forward_pe":              round(forward_pe,  2) if forward_pe  else None,
                "revenue_growth_5yr":      rev_growth,
                "debt_to_equity":          round(debt_eq,     2),
                "dividend_yield":          round(div_yield * 100, 2),
                "payout_ratio":            round(payout  * 100, 2),
                "dividend_sustainability": div_score,
                "roe":                     round(roe    * 100, 2),
                "profit_margin":           round(margin * 100, 2),
                "gross_margin":            round(gross_margin * 100, 2),
                "current_ratio":           round(curr_ratio,  2),
                "beta":                    round(beta,        2),
                "market_cap":              market_cap,
                "sector":                  sector,
                "industry":               industry,
                "analyst_target":          round(analyst_tgt, 2) if analyst_tgt else None,
                "current_price":           round(price,       2) if price       else None,
                "analyst_recommendation":  analyst_rec,
                "moat_rating":             moat,
            }
        except Exception as e:
            return {"error": str(e)}

    def _revenue_cagr(self, stock) -> float:
        try:
            fin = stock.financials
            if fin is None or fin.empty:
                return 0.0
            row = next((i for i in fin.index if "revenue" in str(i).lower()), None)
            if row is None:
                return 0.0
            rev = fin.loc[row].dropna()
            if len(rev) < 2:
                return 0.0
            oldest, newest = float(rev.iloc[-1]), float(rev.iloc[0])
            if oldest <= 0:
                return 0.0
            cagr = (newest / oldest) ** (1 / max(len(rev) - 1, 1)) - 1
            return round(cagr * 100, 2)
        except Exception:
            return 0.0

    def _div_sustainability(self, yld, payout, roe, margin) -> int:
        score = 50
        if yld   > 0:    score += 10
        if payout < 0.6: score += 20
        elif payout > 0.9: score -= 20
        if roe   > 0.15: score += 10
        if margin > 0.10: score += 10
        return max(0, min(100, score))

    def _moat(self, margin, roe, mcap, gross_margin, sector) -> float:
        s = 0
        if margin > 0.20: s += 2
        elif margin > 0.10: s += 1
        if roe > 0.20: s += 2
        elif roe > 0.10: s += 1
        if mcap > 100_000_000_000: s += 2
        elif mcap > 10_000_000_000: s += 1
        if gross_margin > 0.50: s += 2
        elif gross_margin > 0.30: s += 1
        if sector in ("Technology", "Healthcare"): s += 1
        return min(10.0, float(s))


# ── Orchestrator ─────────────────────────────────────────────────────────────
class InformationGatherer:
    def __init__(self):
        self.news   = NewsAnalystAgent()
        self.social = SocialMediaAnalystAgent()
        self.market = MarketAnalystAgent()
        self.fund   = FundamentalAnalystAgent()

    def gather(self, ticker: str) -> Dict:
        news         = self.news.get_news(ticker)
        social       = self.social.get_sentiment_score(ticker, news)
        market       = self.market.analyze(ticker)
        fundamentals = self.fund.analyze(ticker)
        return {
            "ticker":           ticker,
            "timestamp":        datetime.utcnow().isoformat(),
            "news":             news,
            "social_sentiment": social,
            "market":           market,
            "fundamentals":     fundamentals,
        }
