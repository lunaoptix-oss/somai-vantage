"""
Agent 5: Performance Feedback — tracks signal outcomes and feeds performance data back to Agent 1.
"""
import json
import os
from typing import Dict, List
from datetime import datetime


FEEDBACK_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "signal_history.json")


class FeedbackAgent:

    def __init__(self):
        os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
        self.history: List[Dict] = self._load()

    def _load(self) -> List[Dict]:
        if os.path.exists(FEEDBACK_FILE):
            try:
                with open(FEEDBACK_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save(self):
        with open(FEEDBACK_FILE, "w") as f:
            json.dump(self.history, f, indent=2)

    def record_signal(self, execution: Dict, research: Dict):
        """Called when a new signal is generated."""
        record = {
            "id": f"{execution['ticker']}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "ticker": execution["ticker"],
            "action": execution["action"],
            "entry_price": execution.get("entry_price_high"),
            "target_price": execution.get("target_price"),
            "stop_loss": execution.get("stop_loss"),
            "composite_score": research.get("composite_score"),
            "classification": research.get("classification"),
            "generated_at": execution.get("generated_at"),
            "outcome": None,
            "pnl_pct": None,
            "resolved_at": None,
        }
        self.history.append(record)
        self._save()
        return record["id"]

    def resolve_signal(self, signal_id: str, current_price: float):
        """Mark a signal as resolved with current market price."""
        for rec in self.history:
            if rec["id"] == signal_id and rec["outcome"] is None:
                entry = rec.get("entry_price") or 0
                if entry and entry > 0:
                    pnl = (current_price - entry) / entry * 100
                    rec["pnl_pct"] = round(pnl, 2)
                    action = rec.get("action", "")
                    if "BUY" in action:
                        rec["outcome"] = "WIN" if pnl > 0 else "LOSS"
                    elif "SELL" in action:
                        rec["outcome"] = "WIN" if pnl < 0 else "LOSS"
                    else:
                        rec["outcome"] = "NEUTRAL"
                    rec["resolved_at"] = datetime.utcnow().isoformat()
                self._save()
                return rec
        return None

    def get_performance_summary(self) -> Dict:
        resolved = [r for r in self.history if r["outcome"] is not None]
        if not resolved:
            return {
                "total_signals": len(self.history),
                "resolved": 0,
                "win_rate": 0,
                "avg_pnl_pct": 0,
                "best_trade": None,
                "worst_trade": None,
            }

        wins = [r for r in resolved if r["outcome"] == "WIN"]
        pnls = [r["pnl_pct"] for r in resolved if r["pnl_pct"] is not None]
        best = max(resolved, key=lambda x: x.get("pnl_pct") or -999)
        worst = min(resolved, key=lambda x: x.get("pnl_pct") or 999)

        return {
            "total_signals": len(self.history),
            "resolved": len(resolved),
            "win_rate": round(len(wins) / len(resolved) * 100, 1),
            "avg_pnl_pct": round(sum(pnls) / len(pnls), 2) if pnls else 0,
            "best_trade": {"ticker": best["ticker"], "pnl_pct": best.get("pnl_pct")},
            "worst_trade": {"ticker": worst["ticker"], "pnl_pct": worst.get("pnl_pct")},
        }

    def get_agent1_refinements(self) -> Dict:
        """Generates refinement signals for Agent 1 based on outcome patterns."""
        resolved = [r for r in self.history if r["outcome"] is not None]
        if len(resolved) < 5:
            return {"message": "Insufficient data for refinement", "adjustments": {}}

        # Which classifications led to wins?
        from collections import defaultdict
        class_outcomes = defaultdict(list)
        for r in resolved:
            class_outcomes[r.get("classification", "UNKNOWN")].append(
                1 if r["outcome"] == "WIN" else 0
            )

        adjustments = {}
        for cls, outcomes in class_outcomes.items():
            wr = sum(outcomes) / len(outcomes)
            if wr < 0.4:
                adjustments[cls] = f"Lower weight on {cls} signals (win rate: {wr*100:.0f}%)"
            elif wr > 0.7:
                adjustments[cls] = f"Increase confidence in {cls} signals (win rate: {wr*100:.0f}%)"

        return {
            "message": f"Refinements based on {len(resolved)} resolved signals",
            "adjustments": adjustments,
        }
