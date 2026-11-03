"""Report saved test metrics."""
from __future__ import annotations

import json, sys
from config.config import configure_logging
from src.models.evaluator import load_test_metrics

logger = configure_logging(__name__)


def main() -> int:
    metrics = load_test_metrics()
    if not metrics:
        logger.error("No test metrics found. Run scripts/train_model.py first.")
        return 1
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())