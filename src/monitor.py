import argparse
import json
from pathlib import Path

from fifa_churn.modeling import RAW_DATA_PATH, ROOT
from fifa_churn.monitoring import monitor

REPORT_PATH = ROOT / "reports" / "monitoring_report.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor FIFA player feature drift.")
    parser.add_argument(
        "--current-data",
        type=Path,
        default=RAW_DATA_PATH,
        help="Current FIFA CSV batch; defaults to the reference snapshot.",
    )
    args = parser.parse_args()

    report = monitor(args.current_data)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[SUCCESS] Monitoring report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
