"""
Portfolio Engine — paper trading with P&L tracking and evaluation triggers.
Each position records entry price, target, stop-loss, timeframe, and sub-scores
so Agent 5 can evaluate prediction accuracy and retrain weights.
"""
import json, os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import yfinance as yf

PORTFOLIO_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "portfolio.json")

TIMEFRAME_DAYS = {
    "2W":  14,
    "1M":  30,
    "3M":  90,
    "6M": 180,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load() -> List[Dict]:
    try:
        with open(PORTFOLIO_FILE) as f:
            return json.load(f)
    except Exception:
        return []

def _save(positions: List[Dict]):
    os.makedirs(os.path.dirname(PORTFOLIO_FILE), exist_ok=True)
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(positions, f, indent=2)

def _live_price(ticker: str) -> Optional[float]:
    try:
        fi = yf.Ticker(ticker).fast_info
        p  = getattr(fi, "last_price", None)
        return round(float(p), 2) if p else None
    except Exception:
        return None


# ── Portfolio Manager ─────────────────────────────────────────────────────────

class PortfolioManager:

    # ── Add position ──────────────────────────────────────────────────────────
    def add_position(self, ticker: str, analysis: Dict, timeframe: str = "1M") -> Dict:
        """Lock in a stock for monitoring. Returns the new position."""
        positions  = _load()
        fund       = analysis.get("fundamentals", {})
        market     = analysis.get("market", {})
        research   = analysis.get("research", {})
        risk       = analysis.get("risk", {})
        pt         = analysis.get("price_targets", {})
        summary    = analysis.get("summary", {})

        entry_price = pt.get("current") or fund.get("current_price") or market.get("current_price")
        if not entry_price:
            raise ValueError(f"Cannot determine current price for {ticker}")

        days   = TIMEFRAME_DAYS.get(timeframe, 30)
        now    = datetime.utcnow()

        position = {
            "id":              f"{ticker}_{now.strftime('%Y%m%d%H%M%S')}",
            "ticker":          ticker,
            "action":          summary.get("action", "BUY"),
            "classification":  summary.get("classification", "NEUTRAL"),
            "composite_score": summary.get("composite_score", 0.5),
            "sub_scores":      research.get("sub_scores", {}),

            # Prices
            "entry_price":     round(float(entry_price), 2),
            "target_price":    pt.get("bull_target"),
            "analyst_target":  pt.get("analyst_target"),
            "stop_loss":       pt.get("stop_loss"),
            "entry_zone_low":  pt.get("entry_zone_low"),
            "entry_zone_high": pt.get("entry_zone_high"),

            # Context
            "timeframe":       timeframe,
            "target_date":     (now + timedelta(days=days)).isoformat(),
            "added_at":        now.isoformat(),
            "agent_rec":       fund.get("analyst_recommendation", "HOLD"),
            "risk_score":      risk.get("risk_score"),
            "position_size_pct": risk.get("suggested_position_size_pct"),

            # Outcome (filled when evaluated)
            "status":              "OPEN",   # OPEN | WIN | LOSS | STOPPED_OUT | EXPIRED
            "current_price":       None,
            "pnl_pct":             None,
            "pnl_direction":       None,     # UP | DOWN
            "correct_prediction":  None,     # True | False | None
            "evaluated_at":        None,
            "evaluation_notes":    [],
        }

        positions.append(position)
        _save(positions)
        return position

    # ── Get all positions with live prices ────────────────────────────────────
    def get_positions(self, refresh_prices: bool = True) -> List[Dict]:
        positions = _load()
        now       = datetime.utcnow()

        for pos in positions:
            if pos["status"] != "OPEN":
                continue

            # Refresh live price
            if refresh_prices:
                live = _live_price(pos["ticker"])
                if live:
                    pos["current_price"] = live
                    entry = pos["entry_price"]
                    if entry and entry > 0:
                        pnl = round((live - entry) / entry * 100, 2)
                        pos["pnl_pct"]       = pnl
                        pos["pnl_direction"] = "UP" if pnl >= 0 else "DOWN"

            # Check if stop-loss hit
            if pos.get("stop_loss") and pos.get("current_price"):
                if pos["current_price"] <= pos["stop_loss"] and pos["action"] in ("BUY", "STRONG BUY"):
                    pos["status"] = "STOPPED_OUT"

            # Check if target date reached
            if pos.get("target_date"):
                target_dt = datetime.fromisoformat(pos["target_date"])
                pos["days_remaining"] = max(0, (target_dt - now).days)
                pos["days_held"]      = (now - datetime.fromisoformat(pos["added_at"])).days
                pos["due_for_eval"]   = target_dt <= now

        _save(positions)
        return positions

    # ── Close / remove a position ─────────────────────────────────────────────
    def close_position(self, position_id: str) -> Optional[Dict]:
        positions = _load()
        for pos in positions:
            if pos["id"] == position_id and pos["status"] == "OPEN":
                live = _live_price(pos["ticker"])
                if live:
                    pos["current_price"] = live
                    entry = pos["entry_price"]
                    pnl   = round((live - entry) / entry * 100, 2) if entry > 0 else 0
                    pos["pnl_pct"] = pnl
                    pos["status"]  = "WIN" if pnl > 0 else "LOSS"
                else:
                    pos["status"] = "EXPIRED"
                pos["evaluated_at"] = datetime.utcnow().isoformat()
                _save(positions)
                return pos
        return None

    # ── Evaluate a position (Agent 5 trigger) ─────────────────────────────────
    def evaluate_position(self, position_id: str) -> Optional[Dict]:
        """
        Evaluate a matured position. Updates outcome and flags for Agent 5 retraining.
        Returns the evaluated position with correct_prediction set.
        """
        positions = _load()
        for pos in positions:
            if pos["id"] != position_id:
                continue

            live  = _live_price(pos["ticker"]) or pos.get("current_price", pos["entry_price"])
            entry = pos["entry_price"]
            pnl   = round((live - entry) / entry * 100, 2) if entry and entry > 0 else 0

            pos["current_price"] = live
            pos["pnl_pct"]       = pnl
            pos["evaluated_at"]  = datetime.utcnow().isoformat()

            # Was the prediction correct?
            action = pos.get("action", "")
            if "BUY" in action:
                pos["correct_prediction"] = pnl > 0
                pos["status"] = "WIN" if pnl > 0 else "LOSS"
            elif "SELL" in action:
                pos["correct_prediction"] = pnl < 0
                pos["status"] = "WIN" if pnl < 0 else "LOSS"
            else:
                pos["correct_prediction"] = abs(pnl) < 5
                pos["status"] = "EXPIRED"

            # Build evaluation notes
            notes = []
            target = pos.get("target_price")
            if target:
                vs_target = round((live - target) / target * 100, 1)
                notes.append(f"Target was ${target}; reached ${live} ({vs_target:+.1f}% vs target)")
            notes.append(f"P&L: {pnl:+.2f}% over {pos.get('days_held', 0)} days")
            notes.append(f"Prediction {'CORRECT' if pos['correct_prediction'] else 'INCORRECT'}")
            pos["evaluation_notes"] = notes

            _save(positions)
            return pos
        return None

    # ── Portfolio summary ─────────────────────────────────────────────────────
    def get_summary(self) -> Dict:
        positions = _load()
        open_pos  = [p for p in positions if p["status"] == "OPEN"]
        closed    = [p for p in positions if p["status"] in ("WIN", "LOSS", "STOPPED_OUT")]

        total_pnl  = [p["pnl_pct"] for p in closed if p.get("pnl_pct") is not None]
        wins       = [p for p in closed if p["status"] == "WIN"]
        due        = [p for p in open_pos if p.get("due_for_eval")]

        return {
            "open_count":    len(open_pos),
            "closed_count":  len(closed),
            "win_count":     len(wins),
            "loss_count":    len(closed) - len(wins),
            "win_rate":      round(len(wins) / len(closed) * 100, 1) if closed else 0,
            "avg_pnl":       round(sum(total_pnl) / len(total_pnl), 2) if total_pnl else 0,
            "best_trade":    max(closed, key=lambda x: x.get("pnl_pct") or -999, default=None),
            "worst_trade":   min(closed, key=lambda x: x.get("pnl_pct") or 999,  default=None),
            "due_for_eval":  len(due),
            "total_trades":  len(positions),
        }
