"""
Agent 1 — Cloud-safe data gathering for Render deployment.
Strategy: use only endpoints confirmed to work from AWS/Render IPs:
  - yf.download()          → price history, technicals
  - fast_info              → live price, market cap, 52w range
  - t.financials           → revenue CAGR
  - t.balance_sheet        → debt/equity
  - t.income_stmt          → margins, EPS → P/E
  - t.analyst_price_targets→ analyst targets (works in 1.4.1)
  - t.recommendations_summary → buy/sell counts (works in 1.4.1)
  - t.news                 → news headlines
  - t.info (best-effort)   → full metadata if available
"""
import yfinance as yf
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

_SESSION: Optional[requests.Session] = None

def _session():
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        s.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://finance.yahoo.com/",
        })
        try: s.get("https://finance.yahoo.com/", timeout=8)
        except: pass
        _SESSION = s
    return _SESSION

def tk(sym): return yf.Ticker(sym, session=_session())

def n(v, d=0.0):
    try:
        x = float(v)
        return d if x != x else x
    except: return d

def pct(v):
    """Normalise a ratio — if > 1 assume it's already in percent form, divide by 100."""
    x = n(v)
    return x / 100 if abs(x) > 1 else x


# ── News ──────────────────────────────────────────────────────────────────────
class NewsAnalystAgent:
    POS = ["beat","surge","soar","record","growth","profit","upgrade","buy","strong",
           "gain","rally","breakout","bullish","exceed","raised","outperform","boost"]
    NEG = ["miss","drop","fall","loss","decline","downgrade","sell","weak","risk",
           "cut","bearish","concern","warn","below","disappoint","layoff","slump","crash"]

    def get_news(self, sym: str) -> List[Dict]:
        try:
            raw = tk(sym).news or []
            out = []
            for item in raw[:10]:
                c     = item.get("content", {}) if isinstance(item.get("content"), dict) else item
                title = c.get("title", item.get("title", ""))
                src   = (c.get("provider", {}) or {}).get("displayName", "Yahoo Finance")
                if not title: continue
                t_lc  = title.lower()
                p = sum(1 for w in self.POS if w in t_lc)
                g = sum(1 for w in self.NEG if w in t_lc)
                out.append({"title": title, "published": str(c.get("pubDate", "")),
                            "sentiment": round((p-g)/(p+g), 2) if (p+g) else 0.0,
                            "source": src})
            return out
        except: return []


# ── Social ────────────────────────────────────────────────────────────────────
class SocialMediaAnalystAgent:
    def get_sentiment_score(self, sym: str, news: List[Dict]) -> Dict:
        if not news: return {"score": 0.0, "volume": 0, "trend": "neutral"}
        scores = [n(x["sentiment"]) for x in news]
        avg    = float(np.mean(scores))
        return {"score": round(avg, 2), "volume": len(news),
                "trend": "bullish" if avg > 0.15 else "bearish" if avg < -0.15 else "neutral"}


# ── Market ────────────────────────────────────────────────────────────────────
class MarketAnalystAgent:
    def analyze(self, sym: str) -> Dict:
        try:
            hist = yf.download(sym, period="6mo", progress=False,
                               auto_adjust=True, session=_session())
            if hist is None or hist.empty:
                return self._fast(sym)

            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)

            close  = hist["Close"].squeeze().dropna()
            volume = hist["Volume"].squeeze().dropna()
            if len(close) < 5: return self._fast(sym)

            cur   = float(close.iloc[-1])
            ma50  = float(close.rolling(50).mean().iloc[-1])
            ma200 = float(close.rolling(min(200, len(close))).mean().iloc[-1])

            delta = close.diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rs    = gain / loss.replace(0, np.nan)
            rsi   = 100.0 - (100.0 / (1.0 + float(rs.iloc[-1])))
            if np.isnan(rsi): rsi = 50.0

            avg_v     = float(volume.tail(30).mean())
            vol_ratio = round(float(volume.tail(5).mean()) / avg_v, 2) if avg_v > 0 else 1.0

            def ret(p): return round((cur / float(close.iloc[-p]) - 1)*100, 2) if len(close) >= p else 0.0

            signal = ("BUY"  if cur > ma50 > ma200 and rsi < 72 else
                      "SELL" if cur < ma50 < ma200 and rsi > 28 else "HOLD")

            return {
                "current_price": round(cur, 2),
                "ma50":          round(ma50, 2),
                "ma200":         round(ma200, 2),
                "rsi":           round(rsi, 2),
                "volume_ratio":  vol_ratio,
                "high_52w":      round(float(close.tail(252).max()), 2),
                "low_52w":       round(float(close.tail(252).min()), 2),
                "return_1m":     ret(22),
                "return_3m":     ret(66),
                "return_6m":     ret(max(len(close)-1, 1)),
                "signal":        signal,
            }
        except: return self._fast(sym)

    def _fast(self, sym: str) -> Dict:
        try:
            fi = tk(sym).fast_info
            p  = n(getattr(fi, "last_price", 0))
            if not p: return {}
            return {"current_price": round(p, 2), "ma50": None, "ma200": None,
                    "rsi": 50.0, "volume_ratio": 1.0,
                    "high_52w": n(getattr(fi, "year_high", None)) or None,
                    "low_52w":  n(getattr(fi, "year_low",  None)) or None,
                    "return_1m": 0.0, "return_3m": 0.0, "return_6m": 0.0,
                    "signal": "HOLD"}
        except: return {}


# ── Fundamentals ──────────────────────────────────────────────────────────────
class FundamentalAnalystAgent:

    def analyze(self, sym: str) -> Dict:
        t    = tk(sym)
        info = {}
        try: info = t.info or {}
        except: pass

        fi = None
        try: fi = t.fast_info
        except: pass

        # ── Price ────────────────────────────────────────────────────────────
        price = n(info.get("currentPrice") or info.get("regularMarketPrice"))
        if not price and fi:
            price = n(getattr(fi, "last_price", 0))

        # ── Market Cap ───────────────────────────────────────────────────────
        mcap = n(info.get("marketCap"))
        if not mcap and fi:
            mcap = n(getattr(fi, "market_cap", 0))

        # ── Analyst target ───────────────────────────────────────────────────
        analyst_tgt = n(info.get("targetMeanPrice") or info.get("targetMedianPrice"))
        if not analyst_tgt:
            try:
                apt = t.analyst_price_targets
                if isinstance(apt, dict):
                    analyst_tgt = n(apt.get("mean") or apt.get("median"))
                elif apt is not None:
                    analyst_tgt = n(getattr(apt, "mean", None))
            except: pass

        # ── Analyst recommendation ───────────────────────────────────────────
        analyst_rec = str(info.get("recommendationKey") or "").upper().replace("_", " ")
        if not analyst_rec or analyst_rec in ("", "NONE"):
            try:
                rs = t.recommendations_summary
                if rs is not None and len(rs) > 0:
                    r   = rs.iloc[0]
                    def g(col): return n(r[col] if col in r.index else 0)
                    sb, b, h, s, ss = g("strongBuy"), g("buy"), g("hold"), g("sell"), g("strongSell")
                    tot = sb+b+h+s+ss
                    if tot > 0:
                        bull = (sb+b)/tot
                        if   sb/tot >= 0.25: analyst_rec = "STRONG BUY"
                        elif bull  >= 0.55:  analyst_rec = "BUY"
                        elif (s+ss)/tot >= 0.4: analyst_rec = "SELL"
                        else:                analyst_rec = "HOLD"
            except: pass

        if not analyst_rec: analyst_rec = "HOLD"

        # ── Sector / Industry ────────────────────────────────────────────────
        sector   = info.get("sector",   "") or self._sector_guess(sym)
        industry = info.get("industry", "Unknown")

        # ── Beta ─────────────────────────────────────────────────────────────
        beta = n(info.get("beta"), 1.0)
        if not beta or beta == 0: beta = 1.0

        # ── Dividend ─────────────────────────────────────────────────────────
        div_yield = pct(info.get("dividendYield") or info.get("trailingAnnualDividendYield"))
        payout    = pct(info.get("payoutRatio"))

        # ── P/E (try info first, calculate from statements if missing) ───────
        fwd_pe = n(info.get("forwardPE") or info.get("trailingPE"))
        if not fwd_pe and price:
            fwd_pe = self._calc_pe(t, price)

        # ── Margins and ROE (try info, then calculate from statements) ────────
        margin   = pct(info.get("profitMargins"))
        gross_m  = pct(info.get("grossMargins"))
        roe      = pct(info.get("returnOnEquity"))

        if not margin or not roe or not gross_m:
            stmt_data = self._from_statements(t, price, mcap)
            if not margin:   margin  = stmt_data.get("profit_margin", 0)
            if not gross_m:  gross_m = stmt_data.get("gross_margin", 0)
            if not roe:      roe     = stmt_data.get("roe", 0)
            if not fwd_pe:   fwd_pe  = stmt_data.get("pe", None)

        # ── Debt/Equity (try info, then balance sheet) ────────────────────────
        debt_eq = pct(info.get("debtToEquity"))
        if not debt_eq:
            debt_eq = self._calc_de(t)

        # ── Revenue CAGR ─────────────────────────────────────────────────────
        rev_growth = self._rev_cagr(t)

        # ── Scores ───────────────────────────────────────────────────────────
        div_score = self._div_score(div_yield, payout, roe, margin)
        moat      = self._moat(margin, roe, mcap, gross_m, sector)
        curr_r    = n(info.get("currentRatio"))

        return {
            # fwd_pe here is forward P/E from info when available, TTM P/E when calculated
            "forward_pe":              round(fwd_pe, 2) if fwd_pe else None,
            "revenue_growth_5yr":      rev_growth,
            "debt_to_equity":          round(debt_eq, 2),
            "dividend_yield":          round(div_yield * 100, 2),
            "payout_ratio":            round(payout   * 100, 2),
            "dividend_sustainability": div_score,
            "roe":                     round(roe    * 100, 2),
            "profit_margin":           round(margin * 100, 2),
            "gross_margin":            round(gross_m * 100, 2),
            "current_ratio":           round(curr_r, 2),
            "beta":                    round(beta, 2),
            "market_cap":              int(mcap) if mcap else None,
            "sector":                  sector,
            "industry":               industry,
            "analyst_target":          round(analyst_tgt, 2) if analyst_tgt else None,
            "current_price":           round(price, 2)       if price       else None,
            "analyst_recommendation":  analyst_rec,
            "moat_rating":             moat,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _from_statements(self, t, price, mcap) -> Dict:
        """Calculate margins, ROE, P/E from financial statements."""
        out = {}
        try:
            inc = t.income_stmt
            if inc is not None and not inc.empty:
                def row(kw):
                    return next((inc.loc[i] for i in inc.index if kw in str(i).lower()), None)

                rev  = row("total revenue")
                gi   = row("gross profit")
                ni   = row("net income")
                if rev is not None and ni is not None:
                    r, ni_ = float(rev.iloc[0]), float(ni.iloc[0])
                    if r > 0:
                        out["profit_margin"] = ni_ / r
                        if gi is not None: out["gross_margin"] = float(gi.iloc[0]) / r
                    # EPS → P/E
                    if mcap and r > 0 and price:
                        shares = mcap / price if price > 0 else 0
                        if shares > 0:
                            eps = ni_ / shares
                            if eps > 0: out["pe"] = round(price / eps, 2)
        except: pass

        try:
            bal = t.balance_sheet
            inc = t.income_stmt
            if bal is not None and inc is not None and not bal.empty and not inc.empty:
                def brow(kw):
                    return next((bal.loc[i] for i in bal.index if kw in str(i).lower()), None)
                eq  = brow("stockholders equity") or brow("total equity")
                ni_ = next((inc.loc[i] for i in inc.index if "net income" in str(i).lower()), None)
                if eq is not None and ni_ is not None:
                    e, ni_v = float(eq.iloc[0]), float(ni_.iloc[0])
                    if e > 0: out["roe"] = ni_v / e
        except: pass
        return out

    def _calc_de(self, t) -> float:
        try:
            bal = t.balance_sheet
            if bal is None or bal.empty: return 0.0
            def brow(kw): return next((bal.loc[i] for i in bal.index if kw in str(i).lower()), None)
            debt = brow("total debt") or brow("long term debt")
            eq   = brow("stockholders equity") or brow("total equity")
            if debt is not None and eq is not None:
                d, e = float(debt.iloc[0]), float(eq.iloc[0])
                if e > 0: return round(d / e, 2)
        except: pass
        return 0.0

    def _calc_pe(self, t, price: float) -> Optional[float]:
        try:
            inc = t.income_stmt
            bal = t.balance_sheet
            if inc is None or bal is None or inc.empty or bal.empty: return None
            def irow(kw): return next((inc.loc[i] for i in inc.index if kw in str(i).lower()), None)
            ni = irow("net income")
            if ni is None: return None
            fi = t.fast_info
            shares = n(getattr(fi, "shares", 0))
            if not shares:
                mcap = n(getattr(fi, "market_cap", 0))
                shares = mcap / price if price > 0 else 0
            if shares <= 0: return None
            eps = float(ni.iloc[0]) / shares
            return round(price / eps, 2) if eps > 0 else None
        except: return None

    def _rev_cagr(self, t) -> float:
        try:
            fin = t.financials
            if fin is None or fin.empty: return 0.0
            row = next((i for i in fin.index if "revenue" in str(i).lower()), None)
            if not row: return 0.0
            rev = fin.loc[row].dropna()
            if len(rev) < 2: return 0.0
            o, ne = float(rev.iloc[-1]), float(rev.iloc[0])
            if o <= 0: return 0.0
            return round(((ne/o) ** (1/max(len(rev)-1,1)) - 1)*100, 2)
        except: return 0.0

    def _div_score(self, y, p, roe, m) -> int:
        s = 50
        if y > 0:    s += 10
        if p < 0.6:  s += 20
        elif p > 0.9: s -= 20
        if roe > 0.15: s += 10
        if m   > 0.10: s += 10
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

    def _sector_guess(self, sym: str) -> str:
        MAP = {
            "AAPL":"Technology","MSFT":"Technology","NVDA":"Technology","AMZN":"Consumer Cyclical",
            "GOOGL":"Technology","META":"Technology","TSLA":"Consumer Cyclical","AVGO":"Technology",
            "JPM":"Financial Services","V":"Financial Services","UNH":"Healthcare","LLY":"Healthcare",
            "XOM":"Energy","MA":"Financial Services","HD":"Consumer Cyclical","ORCL":"Technology",
            "COST":"Consumer Defensive","AMD":"Technology","CRM":"Technology","NFLX":"Technology",
        }
        return MAP.get(sym, "Technology")


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
