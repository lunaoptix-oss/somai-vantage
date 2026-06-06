"""
Agent 1: Information Gathering
Uses only Yahoo Finance endpoints that work from cloud servers:
  - fast_info        → real-time price, market cap, 52w range
  - analyst_price_targets → mean/high/low analyst targets
  - recommendations_summary → strongBuy/buy/hold/sell counts
  - history()        → OHLCV for technicals
  - financials       → revenue for CAGR
  - news             → headlines + sentiment
"""
import yfinance as yf
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

# ── Session with browser headers ──────────────────────────────────────────────
_SESSION: Optional[requests.Session] = None

def _session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://finance.yahoo.com/",
        })
        try: _SESSION.get("https://finance.yahoo.com/", timeout=8)
        except: pass
    return _SESSION

def ticker(symbol: str) -> yf.Ticker:
    return yf.Ticker(symbol, session=_session())

def n(val, default=0.0):
    try:
        v = float(val)
        return default if (v != v) else v
    except: return default


# ── News Analyst ──────────────────────────────────────────────────────────────
class NewsAnalystAgent:
    POS = ["beat","surge","soar","record","growth","profit","upgrade","buy","strong",
           "gain","rally","breakout","bullish","exceed","raised","outperform","boost"]
    NEG = ["miss","drop","fall","loss","decline","downgrade","sell","weak","risk",
           "cut","bearish","concern","warn","below","disappoint","layoff","slump"]

    def get_news(self, sym: str) -> List[Dict]:
        try:
            news = ticker(sym).news or []
            out  = []
            for item in news[:10]:
                c     = item.get("content", {}) if isinstance(item.get("content"), dict) else item
                title = c.get("title", item.get("title", ""))
                src   = (c.get("provider", {}) or {}).get("displayName", "Yahoo Finance")
                if title:
                    out.append({"title": title,
                                "published": str(c.get("pubDate", "")),
                                "sentiment": self._score(title),
                                "source": src})
            return out
        except: return []

    def _score(self, text: str) -> float:
        t   = text.lower()
        pos = sum(1 for w in self.POS if w in t)
        neg = sum(1 for w in self.NEG if w in t)
        tot = pos + neg
        return round((pos - neg) / tot, 2) if tot else 0.0


# ── Social Analyst ────────────────────────────────────────────────────────────
class SocialMediaAnalystAgent:
    def get_sentiment_score(self, sym: str, news: List[Dict]) -> Dict:
        if not news:
            return {"score": 0.0, "volume": 0, "trend": "neutral"}
        scores = [n(x["sentiment"]) for x in news]
        avg    = float(np.mean(scores))
        trend  = "bullish" if avg > 0.15 else "bearish" if avg < -0.15 else "neutral"
        return {"score": round(avg, 2), "volume": len(news), "trend": trend}


# ── Market Analyst ────────────────────────────────────────────────────────────
class MarketAnalystAgent:
    def analyze(self, sym: str) -> Dict:
        try:
            # yf.download() is the most reliable cloud-safe endpoint
            hist = yf.download(sym, period="6mo", progress=False, auto_adjust=True)

            # Price from fast_info if history empty
            if hist is None or hist.empty:
                fi      = t.fast_info
                current = n(getattr(fi, "last_price", None))
                if not current:
                    return {}
                return {"current_price": round(current, 2),
                        "ma50": None, "ma200": None, "rsi": 50.0,
                        "volume_ratio": 1.0, "high_52w": None, "low_52w": None,
                        "return_1m": 0.0, "return_3m": 0.0, "return_6m": 0.0,
                        "signal": "HOLD"}

            # Handle both single and multi-ticker download column structures
            close  = hist["Close"] if "Close" in hist.columns else hist.xs("Close", axis=1, level=0)
            volume = hist["Volume"] if "Volume" in hist.columns else hist.xs("Volume", axis=1, level=0)
            if hasattr(close, 'squeeze'): close  = close.squeeze()
            if hasattr(volume, 'squeeze'): volume = volume.squeeze()
            close  = close.dropna()
            volume = volume.dropna()
            current = float(close.iloc[-1])

            ma50  = float(close.rolling(50).mean().iloc[-1])
            ma200 = float(close.rolling(min(200, len(close))).mean().iloc[-1])

            delta = close.diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rs    = gain / loss.replace(0, np.nan)
            rsi   = 100 - (100 / (1 + float(rs.iloc[-1])))
            if np.isnan(rsi): rsi = 50.0

            avg_vol   = float(volume.tail(30).mean())
            vol_ratio = round(float(volume.tail(5).mean()) / avg_vol, 2) if avg_vol > 0 else 1.0

            def ret(n_periods):
                return round((current / float(close.iloc[-n_periods]) - 1) * 100, 2) if len(close) >= n_periods else 0.0

            signal = ("BUY"  if current > ma50 > ma200 and rsi < 72 else
                      "SELL" if current < ma50 < ma200 and rsi > 28 else "HOLD")

            return {
                "current_price": round(current, 2),
                "ma50":          round(ma50, 2),
                "ma200":         round(ma200, 2),
                "rsi":           round(rsi, 2),
                "volume_ratio":  vol_ratio,
                "high_52w":      round(float(close.tail(252).max()), 2),
                "low_52w":       round(float(close.tail(252).min()), 2),
                "return_1m":     ret(22),
                "return_3m":     ret(66),
                "return_6m":     ret(max(len(close) - 1, 1)),
                "signal":        signal,
            }
        except Exception as e:
            return {"error": str(e)}


# ── Fundamental Analyst ───────────────────────────────────────────────────────
class FundamentalAnalystAgent:
    def analyze(self, sym: str) -> Dict:
        try:
            t  = ticker(sym)
            fi = t.fast_info           # always works from cloud
            info = {}
            try: info = t.info or {}   # may fail on cloud - that's ok
            except: pass

            # Price — fast_info is reliable
            price = n(getattr(fi, "last_price", None)) or n(info.get("currentPrice")) or n(info.get("regularMarketPrice"))

            # Market cap
            mcap = n(getattr(fi, "market_cap", None)) or n(info.get("marketCap"))

            # Analyst targets — analyst_price_targets works from cloud
            analyst_tgt = None
            analyst_rec = "HOLD"
            try:
                apt = t.analyst_price_targets
                if apt and isinstance(apt, dict):
                    analyst_tgt = n(apt.get("mean") or apt.get("median"))
                elif apt is not None:
                    analyst_tgt = n(getattr(apt, "mean", None))
            except: pass

            # Analyst recommendation from recommendations_summary
            try:
                rs = t.recommendations_summary
                if rs is not None and len(rs) > 0:
                    row        = rs.iloc[0]
                    strong_buy = n(row.get("strongBuy", 0))
                    buy        = n(row.get("buy", 0))
                    hold       = n(row.get("hold", 0))
                    sell       = n(row.get("sell", 0))
                    s_sell     = n(row.get("strongSell", 0))
                    total      = strong_buy + buy + hold + sell + s_sell
                    if total > 0:
                        bull_pct = (strong_buy + buy) / total
                        if   bull_pct >= 0.7: analyst_rec = "STRONG BUY"
                        elif bull_pct >= 0.5: analyst_rec = "BUY"
                        elif (sell + s_sell) / total >= 0.4: analyst_rec = "SELL"
                        else: analyst_rec = "HOLD"
            except: pass

            # Fallback rec from info
            if analyst_rec == "HOLD" and info.get("recommendationKey"):
                analyst_rec = str(info["recommendationKey"]).upper().replace("_", " ")

            # Fundamentals — try info, graceful on failure
            forward_pe   = n(info.get("forwardPE") or info.get("trailingPE"))
            debt_eq      = n(info.get("debtToEquity"))
            if debt_eq > 10: debt_eq /= 100   # normalise if given as percentage
            roe          = n(info.get("returnOnEquity"))
            if abs(roe) > 1: roe /= 100
            margin       = n(info.get("profitMargins"))
            if abs(margin) > 1: margin /= 100
            gross_margin = n(info.get("grossMargins"))
            if abs(gross_margin) > 1: gross_margin /= 100
            beta         = n(info.get("beta"), 1.0)
            div_yield    = n(info.get("dividendYield") or info.get("trailingAnnualDividendYield"))
            if abs(div_yield) > 1: div_yield /= 100
            payout       = n(info.get("payoutRatio"))
            if abs(payout) > 1: payout /= 100
            curr_ratio   = n(info.get("currentRatio"))
            sector       = info.get("sector",   "Technology")
            industry     = info.get("industry", "Unknown")

            rev_growth   = self._rev_cagr(t)
            div_score    = self._div_score(div_yield, payout, roe, margin)
            moat         = self._moat(margin, roe, mcap, gross_margin, sector)

            return {
                "forward_pe":              round(forward_pe, 2) if forward_pe else None,
                "revenue_growth_5yr":      rev_growth,
                "debt_to_equity":          round(debt_eq, 2),
                "dividend_yield":          round(div_yield * 100, 2),
                "payout_ratio":            round(payout   * 100, 2),
                "dividend_sustainability": div_score,
                "roe":                     round(roe      * 100, 2),
                "profit_margin":           round(margin   * 100, 2),
                "gross_margin":            round(gross_margin * 100, 2),
                "current_ratio":           round(curr_ratio, 2),
                "beta":                    round(beta, 2),
                "market_cap":              int(mcap) if mcap else None,
                "sector":                  sector,
                "industry":               industry,
                "analyst_target":          round(analyst_tgt, 2) if analyst_tgt else None,
                "current_price":           round(price, 2) if price else None,
                "analyst_recommendation":  analyst_rec,
                "moat_rating":             moat,
            }
        except Exception as e:
            return {"error": str(e)}

    def _rev_cagr(self, t) -> float:
        try:
            fin = t.financials
            if fin is None or fin.empty: return 0.0
            row = next((i for i in fin.index if "revenue" in str(i).lower()), None)
            if not row: return 0.0
            rev = fin.loc[row].dropna()
            if len(rev) < 2: return 0.0
            oldest, newest = float(rev.iloc[-1]), float(rev.iloc[0])
            if oldest <= 0: return 0.0
            return round(((newest / oldest) ** (1 / max(len(rev)-1,1)) - 1) * 100, 2)
        except: return 0.0

    def _div_score(self, yld, payout, roe, margin) -> int:
        s = 50
        if yld   > 0:    s += 10
        if payout < 0.6: s += 20
        elif payout > 0.9: s -= 20
        if roe   > 0.15: s += 10
        if margin > 0.10: s += 10
        return max(0, min(100, s))

    def _moat(self, margin, roe, mcap, gross_m, sector) -> float:
        s = 0
        if margin > 0.20: s += 2
        elif margin > 0.10: s += 1
        if roe > 0.20: s += 2
        elif roe > 0.10: s += 1
        if mcap > 100e9: s += 2
        elif mcap > 10e9: s += 1
        if gross_m > 0.50: s += 2
        elif gross_m > 0.30: s += 1
        if sector in ("Technology", "Healthcare"): s += 1
        return min(10.0, float(s))


# ── Orchestrator ──────────────────────────────────────────────────────────────
class InformationGatherer:
    def __init__(self):
        self.news_agent   = NewsAnalystAgent()
        self.social_agent = SocialMediaAnalystAgent()
        self.market_agent = MarketAnalystAgent()
        self.fund_agent   = FundamentalAnalystAgent()

    def gather(self, sym: str) -> Dict:
        news = self.news_agent.get_news(sym)
        return {
            "ticker":           sym,
            "timestamp":        datetime.utcnow().isoformat(),
            "news":             news,
            "social_sentiment": self.social_agent.get_sentiment_score(sym, news),
            "market":           self.market_agent.analyze(sym),
            "fundamentals":     self.fund_agent.analyze(sym),
        }
