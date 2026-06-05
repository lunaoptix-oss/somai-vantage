"""
Agent 1: Information Gathering
Uses direct Yahoo Finance API with cookie/crumb auth to work on cloud servers.
"""
import yfinance as yf
import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

# ─── Yahoo Finance session with cookie + crumb (bypasses cloud IP blocks) ────

_SESSION: Optional[requests.Session] = None
_CRUMB:   Optional[str] = None

def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer":         "https://finance.yahoo.com/",
        })
        # Seed cookies
        try:
            _SESSION.get("https://finance.yahoo.com/", timeout=10)
        except Exception:
            pass
    return _SESSION

def _get_crumb() -> Optional[str]:
    global _CRUMB
    if _CRUMB:
        return _CRUMB
    try:
        s = _get_session()
        r = s.get("https://query2.finance.yahoo.com/v1/test/getcrumb", timeout=10)
        if r.status_code == 200 and r.text not in ("", "null"):
            _CRUMB = r.text.strip()
    except Exception:
        pass
    return _CRUMB

def _yf_summary(ticker: str) -> Dict:
    """Fetch quote summary directly from Yahoo Finance API."""
    s     = _get_session()
    crumb = _get_crumb()
    modules = "summaryDetail,defaultKeyStatistics,financialData,recommendationTrend,upgradeDowngradeHistory"
    params  = {"modules": modules, "formatted": "false", "lang": "en-US"}
    if crumb:
        params["crumb"] = crumb

    for host in ("query1", "query2"):
        try:
            url = f"https://{host}.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
            r   = s.get(url, params=params, timeout=15)
            if r.status_code == 200:
                data = r.json()
                result = data.get("quoteSummary", {}).get("result", [])
                if result:
                    return result[0]
        except Exception:
            continue
    return {}

def _yf_quote(ticker: str) -> Dict:
    """Fast quote endpoint — price, change, volume."""
    s = _get_session()
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        r   = s.get(url, params={"interval": "1d", "range": "1d"}, timeout=10)
        if r.status_code == 200:
            meta = r.json().get("chart", {}).get("result", [{}])[0].get("meta", {})
            return meta
    except Exception:
        pass
    return {}

def get_yf_ticker(symbol: str):
    return yf.Ticker(symbol, session=_get_session())


# ─── News Analyst ─────────────────────────────────────────────────────────────
class NewsAnalystAgent:
    POS = ["beat","surge","soar","record","growth","profit","upgrade","buy","strong",
           "gain","rally","breakout","bullish","exceed","raised","outperform","jump","boost"]
    NEG = ["miss","drop","fall","loss","decline","downgrade","sell","weak","risk",
           "cut","bearish","concern","warn","below","disappoint","layoff","slump","crash"]

    def get_news(self, ticker: str) -> List[Dict]:
        try:
            stock = get_yf_ticker(ticker)
            news  = stock.news or []
            out   = []
            for item in news[:10]:
                c     = item.get("content", {}) if isinstance(item.get("content"), dict) else item
                title = c.get("title", item.get("title", ""))
                pub   = c.get("pubDate", item.get("providerPublishTime", ""))
                src   = (c.get("provider", {}) or {}).get("displayName", "Yahoo Finance")
                if title:
                    out.append({
                        "title":     title,
                        "published": str(pub),
                        "sentiment": self._score(title),
                        "source":    src,
                    })
            return out
        except Exception:
            return []

    def _score(self, text: str) -> float:
        t   = text.lower()
        pos = sum(1 for w in self.POS if w in t)
        neg = sum(1 for w in self.NEG if w in t)
        tot = pos + neg
        return round((pos - neg) / tot, 2) if tot else 0.0


# ─── Social Analyst ───────────────────────────────────────────────────────────
class SocialMediaAnalystAgent:
    def get_sentiment_score(self, ticker: str, news: List[Dict]) -> Dict:
        if not news:
            return {"score": 0.0, "volume": 0, "trend": "neutral"}
        scores = [n["sentiment"] for n in news]
        avg    = float(np.mean(scores))
        trend  = "bullish" if avg > 0.15 else ("bearish" if avg < -0.15 else "neutral")
        return {"score": round(avg, 2), "volume": len(news), "trend": trend}


# ─── Market Analyst ───────────────────────────────────────────────────────────
class MarketAnalystAgent:
    def analyze(self, ticker: str) -> Dict:
        try:
            stock = get_yf_ticker(ticker)
            hist  = stock.history(period="6mo")

            # Fallback: get current price from chart API if history empty
            current = None
            if hist.empty:
                meta    = _yf_quote(ticker)
                current = meta.get("regularMarketPrice") or meta.get("previousClose")
                if not current:
                    return {}
                return {
                    "current_price": round(float(current), 2),
                    "ma50": None, "ma200": None,
                    "rsi": 50.0, "volume_ratio": 1.0,
                    "high_52w": None, "low_52w": None,
                    "return_1m": 0.0, "return_3m": 0.0, "return_6m": 0.0,
                    "signal": "HOLD",
                }

            close   = hist["Close"]
            volume  = hist["Volume"]
            current = float(close.iloc[-1])

            ma50  = float(close.rolling(50).mean().iloc[-1])
            ma200 = float(close.rolling(min(200, len(close))).mean().iloc[-1])

            delta = close.diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rs    = gain / loss.replace(0, np.nan)
            rsi   = float(100 - (100 / (1 + rs.iloc[-1])))
            if np.isnan(rsi):
                rsi = 50.0

            avg_vol   = float(volume.tail(30).mean())
            vol_ratio = round(float(volume.tail(5).mean()) / avg_vol, 2) if avg_vol > 0 else 1.0

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
                "high_52w":      round(float(close.tail(252).max()), 2),
                "low_52w":       round(float(close.tail(252).min()), 2),
                "return_1m":     ret(22),
                "return_3m":     ret(66),
                "return_6m":     ret(max(len(close) - 1, 1)),
                "signal":        signal,
            }
        except Exception as e:
            return {"error": str(e)}


# ─── Fundamental Analyst ──────────────────────────────────────────────────────
class FundamentalAnalystAgent:
    def analyze(self, ticker: str) -> Dict:
        try:
            # Primary: direct Yahoo Finance summary API
            summary = _yf_summary(ticker)
            fd  = summary.get("financialData", {})
            sd  = summary.get("summaryDetail", {})
            ks  = summary.get("defaultKeyStatistics", {})
            rec = summary.get("recommendationTrend", {})

            def v(d, *keys):
                for k in keys:
                    val = d.get(k)
                    if isinstance(val, dict):
                        val = val.get("raw", val.get("fmt"))
                    if val is not None:
                        try: return float(val)
                        except: return val
                return None

            price        = v(fd, "currentPrice") or v(sd, "previousClose", "regularMarketPreviousClose")
            forward_pe   = v(sd, "forwardPE") or v(ks, "forwardPE")
            analyst_tgt  = v(fd, "targetMeanPrice", "targetMedianPrice")
            analyst_rec  = fd.get("recommendationKey", "hold")
            if isinstance(analyst_rec, dict):
                analyst_rec = analyst_rec.get("fmt", "hold")
            analyst_rec = str(analyst_rec).upper().replace("_", " ")

            debt_eq      = v(ks, "debtToEquity")
            roe          = v(fd, "returnOnEquity")
            margin       = v(fd, "profitMargins")
            gross_margin = v(fd, "grossProfits")  # will normalise below
            beta         = v(ks, "beta")
            market_cap   = v(sd, "marketCap") or v(ks, "enterpriseValue")
            div_yield    = v(sd, "dividendYield", "trailingAnnualDividendYield") or 0
            payout       = v(sd, "payoutRatio") or 0
            curr_ratio   = v(fd, "currentRatio") or 0

            # Fallback to yfinance .info if summary missing key fields
            if not price or not forward_pe:
                try:
                    info         = get_yf_ticker(ticker).fast_info
                    price        = price or getattr(info, "last_price", None)
                    market_cap   = market_cap or getattr(info, "market_cap", None)
                except Exception:
                    pass

            sector, industry = self._sector(ticker)
            rev_growth = self._revenue_cagr(ticker)

            # Normalise percentages
            if roe   and abs(roe)   > 1: roe   /= 100
            if margin and abs(margin) > 1: margin /= 100
            if div_yield and abs(div_yield) > 1: div_yield /= 100
            if payout and abs(payout) > 1: payout /= 100
            if debt_eq and abs(debt_eq) > 10: debt_eq /= 100

            gross_m = v(fd, "grossMargins") or 0
            if gross_m and abs(gross_m) > 1: gross_m /= 100

            div_score = self._div_score(div_yield, payout, roe or 0, margin or 0)
            moat      = self._moat(margin or 0, roe or 0, market_cap or 0, gross_m, sector)

            return {
                "forward_pe":              round(forward_pe, 2) if forward_pe else None,
                "revenue_growth_5yr":      rev_growth,
                "debt_to_equity":          round(debt_eq / 100, 2) if debt_eq and debt_eq > 1 else round(debt_eq or 0, 2),
                "dividend_yield":          round((div_yield or 0) * 100, 2),
                "payout_ratio":            round((payout or 0)    * 100, 2),
                "dividend_sustainability": div_score,
                "roe":                     round((roe    or 0)    * 100, 2),
                "profit_margin":           round((margin or 0)    * 100, 2),
                "gross_margin":            round(gross_m          * 100, 2),
                "current_ratio":           round(curr_ratio,  2),
                "beta":                    round(beta or 1.0, 2),
                "market_cap":              int(market_cap) if market_cap else None,
                "sector":                  sector,
                "industry":               industry,
                "analyst_target":          round(analyst_tgt, 2) if analyst_tgt else None,
                "current_price":           round(price, 2) if price else None,
                "analyst_recommendation":  analyst_rec,
                "moat_rating":             moat,
            }
        except Exception as e:
            return {"error": str(e)}

    def _sector(self, ticker: str):
        try:
            info = get_yf_ticker(ticker).info
            return info.get("sector", "Unknown"), info.get("industry", "Unknown")
        except Exception:
            return "Unknown", "Unknown"

    def _revenue_cagr(self, ticker: str) -> float:
        try:
            fin = get_yf_ticker(ticker).financials
            if fin is None or fin.empty: return 0.0
            row = next((i for i in fin.index if "revenue" in str(i).lower()), None)
            if not row: return 0.0
            rev = fin.loc[row].dropna()
            if len(rev) < 2: return 0.0
            oldest, newest = float(rev.iloc[-1]), float(rev.iloc[0])
            if oldest <= 0: return 0.0
            return round(((newest / oldest) ** (1 / max(len(rev) - 1, 1)) - 1) * 100, 2)
        except Exception:
            return 0.0

    def _div_score(self, yld, payout, roe, margin) -> int:
        s = 50
        if yld   and yld   > 0:    s += 10
        if payout and payout < 0.6: s += 20
        elif payout and payout > 0.9: s -= 20
        if roe   > 0.15: s += 10
        if margin > 0.10: s += 10
        return max(0, min(100, s))

    def _moat(self, margin, roe, mcap, gross_m, sector) -> float:
        s = 0
        if margin > 0.20: s += 2
        elif margin > 0.10: s += 1
        if roe > 0.20: s += 2
        elif roe > 0.10: s += 1
        if mcap > 100_000_000_000: s += 2
        elif mcap > 10_000_000_000: s += 1
        if gross_m > 0.50: s += 2
        elif gross_m > 0.30: s += 1
        if sector in ("Technology", "Healthcare"): s += 1
        return min(10.0, float(s))


# ─── Main Gatherer ────────────────────────────────────────────────────────────
class InformationGatherer:
    def __init__(self):
        self.news   = NewsAnalystAgent()
        self.social = SocialMediaAnalystAgent()
        self.market = MarketAnalystAgent()
        self.fund   = FundamentalAnalystAgent()

    def gather(self, ticker: str) -> Dict:
        news         = self.news.get_news(ticker)
        return {
            "ticker":           ticker,
            "timestamp":        datetime.utcnow().isoformat(),
            "news":             news,
            "social_sentiment": self.social.get_sentiment_score(ticker, news),
            "market":           self.market.analyze(ticker),
            "fundamentals":     self.fund.analyze(ticker),
        }
