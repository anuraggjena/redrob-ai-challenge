from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBMISSION_PATH = PROJECT_ROOT / "submission.csv"
GITHUB_REPO = "https://github.com/anuraggjena/redrob-ai-challenge"

st.set_page_config(layout="wide", page_title="Redrob Candidate Ranker")
st.title("Redrob Candidate Intelligence & Ranking Engine")
st.caption("CodeCatalyst — Intelligent Candidate Discovery & Ranking Challenge")

st.markdown(
    """
**Sandbox demo:** browse the submitted top-100 ranking. Full 100K reproduction runs locally via `rank.py` (see below).
"""
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Candidates ranked", "100")
col2.metric("Pool size", "100,000")
col3.metric("Features", "150")
col4.metric("Runtime (local)", "81 sec")

if SUBMISSION_PATH.exists():
    submission = pd.read_csv(SUBMISSION_PATH)
    st.subheader("Top 100 submission")
    st.dataframe(submission, use_container_width=True, height=480)

    st.download_button(
        label="Download submission.csv",
        data=SUBMISSION_PATH.read_text(encoding="utf-8"),
        file_name="submission.csv",
        mime="text/csv",
    )
else:
    st.warning("`submission.csv` not found in the repo root.")

with st.expander("How to reproduce full ranking locally"):
    st.code(
        """pip install -r requirements.txt -r requirements-precompute.txt

python scripts/precompute_all.py \\
  --candidates India_runs_data_and_ai_challenge/candidates.jsonl

python rank.py \\
  --candidates India_runs_data_and_ai_challenge/candidates.jsonl \\
  --out ./submission.csv""",
        language="bash",
    )
    st.markdown(f"Repository: [{GITHUB_REPO}]({GITHUB_REPO})")

artifacts_dir = PROJECT_ROOT / "artifacts"
features_path = artifacts_dir / "features.parquet"
if features_dir_ok := features_path.exists():
    st.divider()
    st.subheader("Live rank (optional)")
    st.caption("Only works when pre-computed `artifacts/` are present on this machine.")

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    uploaded = st.file_uploader("Upload candidates JSON/JSONL", type=["json", "jsonl"])
    top_n = st.slider("Top N", min_value=5, max_value=100, value=20)

    if st.button("Rank uploaded file", type="primary") and uploaded is not None:
        import tempfile
        import time

        from src.pipeline.config_loader import load_job_requirements
        from src.pipeline.orchestrator import RankingPipeline
        from src.submission.csv_writer import write_submission

        suffix = Path(uploaded.name).suffix or ".json"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded.getvalue())
            candidates_path = Path(tmp.name)

        try:
            pipeline = RankingPipeline(artifacts_dir, PROJECT_ROOT / "config")
            start = time.perf_counter()
            rows, profiles_by_id = pipeline.run(candidates_path, top_n=top_n)
            job_requirements = load_job_requirements(PROJECT_ROOT / "config" / "job_requirements.yaml")
            write_submission(
                rows,
                candidates_path.with_suffix(".ranked.csv"),
                profiles_by_id=profiles_by_id,
                job_requirements=job_requirements,
            )
            runtime = time.perf_counter() - start
            st.success(f"Ranking complete in {runtime:.2f} seconds")

            result = pd.DataFrame(
                [
                    {
                        "rank": i + 1,
                        "candidate_id": row["candidate_id"],
                        "score": row["raw_score"],
                    }
                    for i, row in enumerate(
                        sorted(rows, key=lambda r: -r["raw_score"])
                    )
                ]
            )
            st.dataframe(result, use_container_width=True)
        except Exception as exc:
            st.error(f"Ranking failed: {exc}")
        finally:
            candidates_path.unlink(missing_ok=True)
elif not features_dir_ok:
    st.info(
        "Live ranking is disabled here because `artifacts/features.parquet` is not bundled. "
        "Judges can reproduce results with `rank.py` on GitHub."
    )
