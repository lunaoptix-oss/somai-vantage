"""
Agent 1: Information Gathering
yfinance 1.4.1+ works from cloud servers with proper session headers.
Uses t.info (primary) + fast_info + analyst_price_targets as fallbacks.
"""
import yfinance as yf
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

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
            "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer":         "https://finance.yahoo.com/",
        })
        try: _SESSION.get("https://finance.yahoo.com/", timeout=8)
        except: pass
    return _SESSION

def tk(sym: str) -> yf.Ticker:
    return yf.Ticker(sym, session=_session())

def n(v, d=0.0):
    try:
        x = float(v)
        return d if x != x else x
    except: return d


# ── News ─────────────────────────────────────────────────────────────────────
class NewsAnalystAgent:
    POS = ["beat","surge","soar","record","growth","profit","upgrade","buy","strong",
           "gain","rally","breakout","bullish","exceed","raised","outperform","boost"]
    NEG = ["miss","drop","fall","loss","decline","downgrade","sell","weak","risk",
           "cut","bearish","concern","warn","below","disappoint","layoff","slump","crash"]

    def get_news(self, sym: str) -> List[Dict]:
        try:
            news = tk(sym).news or []
            out  = []
            for item in news[:10]:
                c     = item.get("content", {}) if isinstance(item.get("content"), dict) else item
                title = c.get("title", item.get("title", ""))
                src   = (c.get("provider", {}) or {}).get("displayName", "Yahoo Finance")
                if title:
                    t = title.lower()
                    pos = sum(1 for w in self.POS if w in t)
                    neg = sum(1 for w in self.NEG if w in t)
                    tot = pos + neg
                    sent = round((pos - neg) / tot, 2) if tot else 0.0
                    out.append({"title": title, "published": str(c.get("pubDate", "")),
                                "sentiment": sent, "source": src})
            return out
        except: return []


# ── Social ────────────────────────────────────────────────────────────────────
class SocialMediaAnalystAgent:
    def get_sentiment_score(self, sym: str, news: List[Dict]) -> Dict:
        if not news:
            return {"score": 0.0, "volume": 0, "trend": "neutral"}
        scores = [n(x["sentiment"]) for x in news]
        avg    = float(np.mean(scores))
        trend  = "bullish" if avg > 0.15 else "bearish" if avg < -0.15 else "neutral"
        return {"score": round(avg, 2), "volume": len(news), "trend": trend}


# ── Market ────────────────────────────────────────────────────────────────────
class MarketAnalystAgent:
    def analyze(self, sym: str) -> Dict:
        try:
            # yf.download is the most reliable endpoint across all environments
            hist = yf.download(sym, period="6mo", progress=False,
                               auto_adjust=True, session=_session())
            if hist is None or hist.empty:
                return self._from_fast_info(sym)

            # Handle multi-index columns from newer yfinance
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)

            close  = hist["Close"].squeeze().dropna()
            volume = hist["Volume"].squeeze().dropna()

            if len(close) < 5:
                return self._from_fast_info(sym)

            current = float(close.iloc[-1])
            n_close = min(200, len(close))
            ma50  = float(close.rolling(50).mean().iloc[-1])
            ma200 = float(close.rolling(n_close).mean().iloc[-1])

            delta = close.diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rs    = gain / loss.replace(0, np.nan)
            rsi   = 100.0 - (100.0 / (1.0 + float(rs.iloc[-1])))
            if np.isnan(rsi): rsi = 50.0

            avg_vol   = float(volume.tail(30).mean())
            vol_ratio = round(float(volume.tail(5).mean()) / avg_vol, 2) if avg_vol > 0 else 1.0

            def ret(p):
                return round((current / float(close.iloc[-p]) - 1) * 100, 2) if len(close) >= p else 0.0

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
            return self._from_fast_info(sym)

    def _from_fast_info(self, sym: str) -> Dict:
        try:
            fi = tk(sym).fast_info
            price = n(getattr(fi, "last_price", None))
            if not price: return {}
            return {
                "current_price": round(price, 2),
                "ma50": None, "ma200": None, "rsi": 50.0,
                "volume_ratio": 1.0,
                "high_52w": n(getattr(fi, "year_high", None)) or None,
                "low_52w":  n(getattr(fi, "year_low",  None)) or None,
                "return_1m": 0.0, "return_3m": 0.0, "return_6m": 0.0,
                "signal": "HOLD",
            }
        except: return {}


# ── Fundamentals ──────────────────────────────────────────────────────────────
class FundamentalAnalystAgent:
    def analyze(self, sym: str) -> Dict:
        try:
            t    = tk(sym)
            info = {}

            # Try t.info — works in yfinance 1.4.1+ even from cloud
            try:
                info = t.info or {}
            except Exception:
                pass

            # Price: info first, then fast_info
            price = n(info.get("currentPrice") or info.get("regularMarketPrice"))
            if not price:
                try: price = n(getattr(t.fast_info, "last_price", 0))
                except: pass

            # Analyst target: info first, then analyst_price_targets
            analyst_tgt = n(info.get("targetMeanPrice") or info.get("targetMedianPrice"))
            if not analyst_tgt:
                try:
                    apt = t.analyst_price_targets
                    if isinstance(apt, dict):
                        analyst_tgt = n(apt.get("mean") or apt.get("median"))
                    elif apt is not None:
                        analyst_tgt = n(getattr(apt, "mean", None))
                except: pass

            # Recommendation
            analyst_rec = str(info.get("recommendationKey") or "hold").upper().replace("_", " ")
            if analyst_rec in ("", "HOLD", "NONE"):
                try:
                    rs = t.recommendations_summary
                    if rs is not None and len(rs) > 0:
                        row = rs.iloc[0]
                        sb  = n(row.get("strongBuy",  0) if hasattr(row, "get") else row["strongBuy"])
                        b   = n(row.get("buy",         0) if hasattr(row, "get") else row["buy"])
                        h   = n(row.get("hold",        0) if hasattr(row, "get") else row["hold"])
                        s   = n(row.get("sell",        0) if hasattr(row, "get") else row["sell"])
                        ss  = n(row.get("strongSell",  0) if hasattr(row, "get") else row["strongSell"])
                        tot = sb + b + h + s + ss
                        if tot > 0:
                            bull = (sb + b) / tot
                            bear = (s + ss) / tot
                            if   sb / tot >= 0.3:   analyst_rec = "STRONG BUY"
                            elif bull >= 0.6:        analyst_rec = "BUY"
                            elif bear >= 0.4:        analyst_rec = "SELL"
                            else:                    analyst_rec = "HOLD"
                except: pass

            # Market cap
            mcap = n(info.get("marketCap"))
            if not mcap:
                try: mcap = n(getattr(t.fast_info, "market_cap", 0))
                except: pass

            # Core fundamentals
            fwd_pe    = n(info.get("forwardPE")     or info.get("trailingPE"))
            debt_eq   = n(info.get("debtToEquity"))
            if debt_eq > 10: debt_eq /= 100
            roe       = n(info.get("returnOnEquity"))
            if abs(roe) > 1: roe /= 100
            margin    = n(info.get("profitMargins"))
            if abs(margin) > 1: margin /= 100
            gross_m   = n(info.get("grossMargins"))
            if abs(gross_m) > 1: gross_m /= 100
            beta      = n(info.get("beta"), 1.0)
            div_yield = n(info.get("dividendYield") or info.get("trailingAnnualDividendYield"))
            if abs(div_yield) > 1: div_yield /= 100
            payout    = n(info.get("payoutRatio"))
            if abs(payout) > 1: payout /= 100
            curr_r    = n(info.get("currentRatio"))
            sector    = info.get("sector",   "Technology")
            industry  = info.get("industry", "Unknown")

            rev_growth = self._rev_cagr(t)
            div_score  = self._div_score(div_yield, payout, roe, margin)
            moat       = self._moat(margin, roe, mcap, gross_m, sector)

            return {
                "forward_pe":              round(fwd_pe, 2) if fwd_pe else None,
                "revenue_growth_5yr":      rev_growth,
                "debt_to_equity":          round(debt_eq, 2),
                "dividend_yield":          round(div_yield * 100, 2),
                "payout_ratio":            round(payout * 100, 2),
                "dividend_sustainability": div_score,
                "roe":                     round(roe * 100, 2),
                "profit_margin":           round(margin * 100, 2),
                "gross_margin":            round(gross_m * 100, 2),
                "current_ratio":           round(curr_r, 2),
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
            return round(((newest / oldest) ** (1 / max(len(rev)-1, 1)) - 1) * 100, 2)
        except: return 0.0

    def _div_score(self, y, p, roe, m) -> int:
        s = 50
        if y > 0:    s += 10
        if p < 0.6:  s += 20
        elif p > 0.9: s -= 20
        if roe > 0.15: s += 10
        if m > 0.10:   s += 10
        return max(0, min(100, s))

    def _moat(self, m, roe, mcap, gm, sector) -> float:
        s = 0
        if m   > 0.20: s += 2
        elif m > 0.10: s += 1
        if roe > 0.20: s += 2
        elif roe > 0.10: s += 1
        if mcap > 100e9: s += 2
        elif mcap > 10e9: s += 1
        if gm > 0.50:   s += 2
        elif gm > 0.30: s += 1
        if sector in ("Technology", "Healthcare"): s += 1
        return min(10.0, float(s))


# ── Orchestrator ──────────────────────────────────────────────────────────────
class InformationGatherer:
    def __init__(self):
        self.news_a   = NewsAnalystAgent()
        self.social_a = SocialMediaAnalystAgent()
        self.market_a = MarketAnalystAgent()
        self.fund_a   = FundamentalAnalystAgent()

    def gather(self, sym: str) -> Dict:
        news = self.news_a.get_news(sym)
        return {
            "ticker":           sym,
            "timestamp":        datetime.utcnow().isoformat(),
            "news":             news,
            "social_sentiment": self.social_a.get_sentiment_score(sym, news),
            "market":           self.market_a.analyze(sym),
            "fundamentals":     self.fund_a.analyze(sym),
        }
