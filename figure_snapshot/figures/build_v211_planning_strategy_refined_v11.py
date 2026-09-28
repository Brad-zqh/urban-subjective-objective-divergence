"""Final 14-panel JPG/SVG candidate with zero visible-text collisions."""

from pathlib import Path
import build_v211_planning_strategy_refined_v10 as final_layout

final_layout.revised.base.OUT = Path(__file__).resolve().parents[1] / "figures/v211_planning_strategy_refined_v11"

if __name__ == "__main__":
    final_layout.revised.base.main()
