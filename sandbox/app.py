from __future__ import annotations

import io
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.config_loader import load_job_requirements
from src.pipeline.orchestrator import RankingPipeline
from src.submission.csv_writer import write_submission
from src.understanding.parser import load_candidates

st.set_page_config(layout="wide", page_title="Redrob Candidate Ranker Demo")

SAMPLE_PATH = Path(__file__).resolve().parent / "sample_input.json"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CONFIG_DIR = PROJECT_ROOT / "config"
TOP_N = 20

st.title("Redrob Candidate Ranker Demo")

uploaded = st.file_uploader("Upload candidates (JSON or JSONL)", type=["json", "jsonl"])
use_sample = st.button("Run on sample")

if "input_bytes" not in st.session_state:
    st.session_state.input_bytes = None
    st.session_state.input_suffix = ".json"

if use_sample:
    st.session_state.input_bytes = SAMPLE_PATH.read_bytes()
    st.session_state.input_suffix = ".json"
    st.session_state.input_label = str(SAMPLE_PATH.name)

if uploaded is not None:
    st.session_state.input_bytes = uploaded.getvalue()
    suffix = Path(uploaded.name).suffix.lower()
    st.session_state.input_suffix = suffix if suffix in {".json", ".jsonl"} else ".json"
    st.session_state.input_label = uploaded.name

if st.session_state.input_bytes:
    label = st.session_state.get("input_label", "uploaded file")
    st.info(f"Ready to rank: {label}")

rank_clicked = st.button("Rank", type="primary")

if rank_clicked:
    if not st.session_state.input_bytes:
        st.error("Upload a candidates file or click 'Run on sample' first.")
    else:
        with tempfile.NamedTemporaryFile(
            suffix=st.session_state.input_suffix,
            delete=False,
        ) as tmp:
            tmp.write(st.session_state.input_bytes)
            candidates_path = Path(tmp.name)

        try:
            profiles = load_candidates(candidates_path)
            profiles_by_id = {profile.candidate_id: profile for profile in profiles}

            pipeline = RankingPipeline(ARTIFACTS_DIR, CONFIG_DIR)
            start = time.perf_counter()
            rows, profiles_by_id = pipeline.run(candidates_path, top_n=TOP_N)
            job_requirements = load_job_requirements(CONFIG_DIR / "job_requirements.yaml")
            write_submission(
                rows,
                candidates_path.with_suffix(".ranked.csv"),
                profiles_by_id=profiles_by_id,
                job_requirements=job_requirements,
            )
            runtime = time.perf_counter() - start

            features_by_id = pipeline._load_features_from_artifacts(
                [profile.candidate_id for profile in profiles]
            )

            display_rows: list[dict] = []
            for row in sorted(rows, key=lambda r: -r["raw_score"]):
                candidate_id = row["candidate_id"]
                profile = profiles_by_id.get(candidate_id)
                current_title = ""
                if profile is not None:
                    current_title = profile.current_title or profile.raw.get("profile", {}).get(
                        "current_title", ""
                    )

                entry = {
                    "rank": len(display_rows) + 1,
                    "candidate_id": candidate_id,
                    "current_title": current_title,
                    "score": row["raw_score"],
                    "reasoning": "",
                }
                honeypot = features_by_id.get(candidate_id, {}).get("honeypot_probability")
                if honeypot is not None:
                    entry["honeypot_probability"] = float(honeypot)
                display_rows.append(entry)

            df = pd.DataFrame(display_rows)
            column_order = [
                "rank",
                "candidate_id",
                "current_title",
                "score",
                "reasoning",
            ]
            if "honeypot_probability" in df.columns:
                column_order.append("honeypot_probability")
            df = df[column_order]

            st.success(f"Ranking complete in {runtime:.2f} seconds")
            st.dataframe(df, use_container_width=True)

            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download CSV",
                data=csv_buffer.getvalue(),
                file_name="ranked_candidates.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Ranking failed: {exc}")
        finally:
            candidates_path.unlink(missing_ok=True)
