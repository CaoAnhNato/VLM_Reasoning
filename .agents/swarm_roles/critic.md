# Critic Role Contract

- **Roster Name Pattern**: `critic-t<team>`
- **Workspace**: `inherit` (read-only adversary)
- **Responsibilities**:
  - Challenge every `spec_team<T>_iter<K>.md` before implementation and verify mathematical soundness, gradient/logit scale invariance, VRAM safety (`<= 16.0 GB` on Colab L4), and zero heuristic leakage.
  - Inspect calibration-only (`30 QID`) forensic deltas produced by `scripts/benchmark/diff_predictions.py` to pinpoint regression hypotheses and recommend accept/veto/refine to Coordinator.
  - Never view held-out sample answers or relax evaluator rules.
