"""
Agent 5: Cyclic Learning Engine
- Evaluates closed portfolio positions
- Measures which sub-score dimensions predicted outcomes correctly
- Adjusts Agent 2's composite weights via gradient-like updates
- Saves learned weights to data/learned_weights.json
- The loop: Predict → Monitor → Evaluate → Retrain → Predict better
"""
import json, os, math
from datetime import datetime
from typing import Dict, List, Optional

WEIGHTS_FILE  = os.path.join(os.path.dirname(__file__), "..", "data", "learned_weights.json")
HISTORY_FILE  = os.path.join(os.path.dirname(__file__), "..", "data", "signal_history.json")
RETRAIN_LOG   = os.path.join(os.path.dirname(__file__), "..", "data", "retrain_log.json")

# Default weights (must sum to 1.0)
DEFAULT_WEIGHTS = {
    "fundamental_quality": 0.30,
    "technical_signal":    0.20,
    "momentum":            0.15,
    "news_sentiment":      0.15,
    "social_sentiment":    0.10,
    "valuation":           0.10,
}

LEARNING_RATE = 0.05   # How aggressively to shift weights (0.01–0.10)
MIN_WEIGHT    = 0.05   # Floor — no dimension drops below 5%
MAX_WEIGHT    = 0.45   # Ceiling — no dimension dominates above 45%


# ── Weight management ─────────────────────────────────────────────────────────

def load_weights() -> Dict[str, float]:
    """Return current learned weights, or defaults if not yet trained."""
    try:
        with open(WEIGHTS_FILE) as f:
            w = json.load(f)
        # Validate and fill any missing keys
        for k, v in DEFAULT_WEIGHTS.items():
            if k not in w:
                w[k] = v
        return _normalise(w)
    except Exception:
        return dict(DEFAULT_WEIGHTS)

def _normalise(w: Dict) -> Dict:
    """Ensure weights sum to exactly 1.0 and respect min/max bounds."""
    # Clip
    w = {k: max(MIN_WEIGHT, min(MAX_WEIGHT, v)) for k, v in w.items()}
    # Normalise
    total = sum(w.values())
    return {k: round(v / total, 4) for k, v in w.items()}

def _save_weights(w: Dict, reason: str, metrics: Dict):
    os.makedirs(os.path.dirname(WEIGHTS_FILE), exist_ok=True)
    with open(WEIGHTS_FILE, "w") as f:
        json.dump(w, f, indent=2)

    # Append to retrain log
    log = []
    try:
        with open(RETRAIN_LOG) as f:
            log = json.load(f)
    except Exception:
        pass
    log.append({
        "timestamp": datetime.utcnow().isoformat(),
        "reason":    reason,
        "weights":   w,
        "metrics":   metrics,
    })
    with open(RETRAIN_LOG, "w") as f:
        json.dump(log, f, indent=2)


# ── Signal History (legacy) ────────────────────────────────────────────────────

def _load_history() -> List[Dict]:
    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except Exception:
        return []

def _save_history(h: List[Dict]):
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(h, f, indent=2)


# ── Agent 5 Core ──────────────────────────────────────────────────────────────

class FeedbackAgent:

    def __init__(self):
        self.history = _load_history()

    # ── Legacy signal recording (for signal history tab) ─────────────────────
    def record_signal(self, execution: Dict, research: Dict) -> str:
        record = {
            "id":              f"{execution['ticker']}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "ticker":          execution["ticker"],
            "action":          execution["action"],
            "entry_price":     execution.get("entry_price_high"),
            "target_price":    execution.get("target_price"),
            "stop_loss":       execution.get("stop_loss"),
            "composite_score": research.get("composite_score"),
            "classification":  research.get("classification"),
            "sub_scores":      research.get("sub_scores", {}),
            "generated_at":    execution.get("generated_at"),
            "outcome":         None,
            "pnl_pct":         None,
            "resolved_at":     None,
        }
        self.history.append(record)
        _save_history(self.history)
        return record["id"]

    def resolve_signal(self, signal_id: str, current_price: float) -> Optional[Dict]:
        for rec in self.history:
            if rec["id"] == signal_id and rec["outcome"] is None:
                entry = rec.get("entry_price") or 0
                if entry and entry > 0:
                    pnl          = round((current_price - entry) / entry * 100, 2)
                    rec["pnl_pct"] = pnl
                    action         = rec.get("action", "")
                    rec["outcome"] = ("WIN"  if ("BUY"  in action and pnl > 0) or
                                               ("SELL" in action and pnl < 0) else "LOSS")
                    rec["resolved_at"] = datetime.utcnow().isoformat()
                _save_history(self.history)
                return rec
        return None

    # ── Portfolio-based retraining ────────────────────────────────────────────
    def retrain_from_portfolio(self, evaluated_positions: List[Dict]) -> Dict:
        """
        Core learning loop:
        1. For each evaluated position, check which sub-scores predicted correctly
        2. Adjust weights: correct predictors → up, incorrect → down
        3. Normalise and save
        Returns a retraining report.
        """
        if not evaluated_positions:
            return {"message": "No evaluated positions to learn from", "adjustments": {}}

        current_weights = load_weights()
        score_accuracy  = {k: [] for k in current_weights}  # list of 0/1 per signal

        for pos in evaluated_positions:
            correct    = pos.get("correct_prediction")
            sub_scores = pos.get("sub_scores", {})
            if correct is None or not sub_scores:
                continue

            for dim, score in sub_scores.items():
                if dim not in score_accuracy:
                    continue
                # A dimension "contributed correctly" if:
                # - score was bullish (> 0.55) AND prediction was correct, OR
                # - score was bearish (< 0.45) AND prediction was incorrect (meaning we should have listened more)
                bullish_signal = score > 0.55
                if (bullish_signal and correct) or (not bullish_signal and not correct):
                    score_accuracy[dim].append(1)
                else:
                    score_accuracy[dim].append(0)

        if not any(score_accuracy.values()):
            return {"message": "Insufficient sub-score data for retraining", "adjustments": {}}

        # Compute accuracy per dimension
        dim_accuracy = {}
        for dim, outcomes in score_accuracy.items():
            if outcomes:
                dim_accuracy[dim] = sum(outcomes) / len(outcomes)

        # Gradient update: move weight toward accuracy signal
        new_weights = dict(current_weights)
        adjustments = {}
        for dim, acc in dim_accuracy.items():
            old_w  = current_weights[dim]
            # acc > 0.5 means this dimension is predicting well → increase weight
            # acc < 0.5 means it's misleading → decrease weight
            delta  = LEARNING_RATE * (acc - 0.5) * 2   # normalised to [-LR, +LR]
            new_w  = old_w + delta
            new_weights[dim] = new_w
            if abs(delta) > 0.002:
                direction = "up" if delta > 0 else "down"
                adjustments[dim] = {
                    "old":      round(old_w, 4),
                    "new":      round(new_w, 4),
                    "delta":    round(delta, 4),
                    "accuracy": round(acc, 3),
                    "direction": direction,
                    "reason":   f"{dim.replace('_',' ')} was {round(acc*100,0):.0f}% accurate → weight {direction}",
                }

        new_weights = _normalise(new_weights)
        metrics = {
            "positions_evaluated": len(evaluated_positions),
            "wins":  sum(1 for p in evaluated_positions if p.get("status") == "WIN"),
            "losses": sum(1 for p in evaluated_positions if p.get("status") == "LOSS"),
            "dimension_accuracies": {k: round(v, 3) for k, v in dim_accuracy.items()},
        }

        reason = (f"Retrained on {len(evaluated_positions)} positions: "
                  f"{metrics['wins']}W / {metrics['losses']}L")
        _save_weights(new_weights, reason, metrics)

        return {
            "message":      reason,
            "old_weights":  current_weights,
            "new_weights":  new_weights,
            "adjustments":  adjustments,
            "metrics":      metrics,
        }

    # ── Performance summary ────────────────────────────────────────────────────
    def get_performance_summary(self) -> Dict:
        resolved = [r for r in self.history if r.get("outcome")]
        wins     = [r for r in resolved if r["outcome"] == "WIN"]
        pnls     = [r["pnl_pct"] for r in resolved if r.get("pnl_pct") is not None]
        best     = max(resolved, key=lambda x: x.get("pnl_pct") or -999, default=None)
        worst    = min(resolved, key=lambda x: x.get("pnl_pct") or 999,  default=None)
        return {
            "total_signals": len(self.history),
            "resolved":      len(resolved),
            "win_rate":      round(len(wins) / len(resolved) * 100, 1) if resolved else 0,
            "avg_pnl_pct":   round(sum(pnls) / len(pnls), 2) if pnls else 0,
            "best_trade":    {"ticker": best["ticker"],  "pnl_pct": best.get("pnl_pct")}  if best  else None,
            "worst_trade":   {"ticker": worst["ticker"], "pnl_pct": worst.get("pnl_pct")} if worst else None,
        }

    # ── Retrain log ────────────────────────────────────────────────────────────
    def get_retrain_log(self) -> List[Dict]:
        try:
            with open(RETRAIN_LOG) as f:
                return json.load(f)
        except Exception:
            return []

    # ── Weight insight for UI ──────────────────────────────────────────────────
    def get_weight_insight(self) -> Dict:
        weights    = load_weights()
        defaults   = DEFAULT_WEIGHTS
        drift      = {k: round(weights[k] - defaults[k], 4) for k in weights}
        log        = self.get_retrain_log()
        last_train = log[-1] if log else None
        return {
            "current_weights": weights,
            "default_weights": defaults,
            "drift":           drift,
            "retrain_count":   len(log),
            "last_retrained":  last_train["timestamp"] if last_train else None,
            "last_reason":     last_train["reason"]    if last_train else "No retraining yet",
        }

    # ── Legacy ────────────────────────────────────────────────────────────────
    def get_agent1_refinements(self) -> Dict:
        insight = self.get_weight_insight()
        drift   = insight["drift"]
        adj     = {}
        for k, d in drift.items():
            if abs(d) > 0.02:
                adj[k] = f"{'Increased' if d > 0 else 'Decreased'} from default by {abs(d)*100:.1f}pp — higher prediction accuracy detected"
        return {
            "message":     insight["last_reason"],
            "adjustments": adj,
        }
