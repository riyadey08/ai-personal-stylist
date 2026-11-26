"""Smoke test the Streamlit script with a profile present (no API calls)."""

import os
import sys
from pathlib import Path

os.environ["GOOGLE_API_KEY"] = "dummy-key"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest

from src.body_shape import classify
from src.profile import BodyProfile

at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60)
at.run()
if at.exception:
    raise SystemExit(f"initial run failed: {at.exception}")

profile = BodyProfile(
    features={"shoulder_hip_ratio": 1.7},
    shape=classify(1.7),
    measurements={},
    stability={},
    n_photos=1,
)
at.session_state["pipeline"].profile = profile
at.run()
if at.exception:
    raise SystemExit(f"profile run failed: {at.exception}")

# Exercise the rating submission once, then remove the generated dummy row.
rating_file = Path(__file__).resolve().parents[1] / "data" / "dress_ratings.csv"
existed = rating_file.exists()
old_bytes = rating_file.read_bytes() if existed else b""
submit = [b for b in at.button if b.label == "Submit rating"]
if not submit:
    raise SystemExit("Submit rating button not found")
submit[0].click().run()
if at.exception:
    raise SystemExit(f"rating click failed: {at.exception}")
import pandas as pd

assert len(pd.read_csv(rating_file)) == 1
if existed:
    rating_file.write_bytes(old_bytes)
else:
    rating_file.unlink(missing_ok=True)
print("AppTest OK; tabs:", len(at.tabs), "| rating submission OK")
