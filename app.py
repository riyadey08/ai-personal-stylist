"""Streamlit UI for the AI Personal Stylist.

Run: streamlit run app.py
Requires GOOGLE_API_KEY in a .env file (see .env.example).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import config
from src import labeling
from src.dress import DressAnalysis, analyze_dress_cv, decode_image, score_fit
from src.evaluation import (
    body_shape_metrics,
    dress_fit_metrics,
    feature_parity,
    retrieval_metrics,
    write_report,
)
from src.knowledge import FashionKnowledgeBase
from src.llm import GeminiClient, MissingAPIKeyError
from src.pipeline import StylistPipeline
from src.pose import PoseEstimator
from src.retrieval import StyleIndex

st.set_page_config(page_title="AI Personal Stylist", page_icon="👗", layout="wide")


@st.cache_resource(show_spinner="Loading pose model and FAISS index...")
def resources():
    config.ensure_dirs()
    if not config.INDEX_PATH.exists():
        StyleIndex.build(config.TRAINING_DATA_PATH, config.MODELS_DIR)
    return PoseEstimator(), StyleIndex.load(), FashionKnowledgeBase()


try:
    llm = GeminiClient.from_env()
except MissingAPIKeyError as exc:
    st.error(f"{exc} The app requires a Gemini API key by design.")
    st.stop()

estimator, style_index, knowledge = resources()

if "pipeline" not in st.session_state:
    st.session_state.pipeline = StylistPipeline(estimator, style_index, knowledge, llm)
pipeline: StylistPipeline = st.session_state.pipeline

st.title("AI Personal Stylist")
st.caption("Photo(s) + occasion → body profile → outfit advice, plus a dress-fit verdict.")

with st.sidebar:
    st.subheader("Session")
    st.write(f"Model: `{config.model_name()}`")
    st.write("API key: loaded ✓")
    occasion = st.selectbox("Occasion", config.OCCASIONS)
    season = st.selectbox("Season", config.SEASONS)
    st.divider()
    st.caption(
        "Body shape and fit scores are explainable heuristics grounded in the "
        "fashion knowledge base, not a trained medical or fit model."
    )
    if st.button("Reset body profile"):
        pipeline.reset()
        st.rerun()

tab_profile, tab_outfit, tab_dress, tab_eval, tab_label = st.tabs(
    [
        "1 · Body profile",
        "2 · Outfit recommendation",
        "3 · Dress check",
        "4 · Evaluation",
        "5 · Collect ratings",
    ]
)

# ---------------------------------------------------------------------------
with tab_profile:
    st.header("Build a body profile from your photos")
    uploads = st.file_uploader(
        "Upload one or more full-body photos (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="body_photos",
    )
    names = [f.name for f in uploads] if uploads else []
    fitted = st.multiselect(
        "Which photos show fitted clothes? (they refine shoulder/hip width)",
        options=names,
        key="fitted_photos",
    )
    if uploads and st.button("Analyse photos", type="primary"):
        pipeline.reset()
        for upload in uploads:
            error, annotated = pipeline.add_photo(
                upload.getvalue(), fitted=upload.name in fitted, name=upload.name
            )
            if error:
                st.warning(f"{upload.name}: {error}")
            elif annotated is not None:
                st.image(annotated, caption=f"{upload.name} · pose detected", use_container_width=True)
        if pipeline.observations:
            st.success(f"{len(pipeline.observations)} usable photo(s). Now build the profile.")

    if pipeline.observations and st.button("Build profile", type="primary"):
        profile = pipeline.make_profile()
        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated body shape", profile.shape.shape)
        c2.metric("Confidence (heuristic)", f"{profile.shape.confidence:.2f}")
        c3.metric("Shoulder/hip ratio", f"{profile.shape.ratio:.2f}")
        st.write(
            f"Photos: {profile.n_photos} · fitted: {profile.n_fitted} · "
            f"independent agreement: {profile.agreement:.0%}"
        )
        st.caption(profile.shape.explanation)
        st.subheader("Measurements")
        st.dataframe(
            {"feature": list(profile.measurements), "value": list(profile.measurements.values())},
            use_container_width=True,
        )
        st.subheader("Cross-photo stability")
        st.dataframe(profile.stability, use_container_width=True)
        with st.expander("Evidence sent to Gemini"):
            st.code(profile.to_evidence_text())

# ---------------------------------------------------------------------------
with tab_outfit:
    st.header("Outfit recommendation")
    if pipeline.profile is None:
        st.info("Build a body profile in tab 1 first.")
    elif st.button("Generate recommendation", type="primary"):
        with st.spinner("Retrieving similar profiles and asking Gemini..."):
            result = pipeline.recommend(occasion=occasion, season=season)
        st.markdown(result["advice"])
        with st.expander("Similar body profiles retrieved (FAISS)"):
            st.dataframe(result["retrieved"], use_container_width=True)
        with st.expander("Fashion rules used as grounding"):
            st.dataframe(result["kb_rows"], use_container_width=True)

# ---------------------------------------------------------------------------
with tab_dress:
    st.header("Will this dress look flawless on you?")
    if pipeline.profile is None:
        st.info("Build a body profile in tab 1 first.")
    else:
        dress_upload = st.file_uploader("Upload a dress photo", type=["jpg", "jpeg", "png"], key="dress_photo")
        if dress_upload and st.button("Evaluate dress", type="primary"):
            with st.spinner("Analysing the dress and comparing with your profile..."):
                result = pipeline.check_dress(
                    dress_upload.getvalue(), name=dress_upload.name, occasion=occasion, season=season
                )
            verdict = result["verdict"]
            col1, col2 = st.columns([1, 2])
            col1.image(result["image"], caption="Analysed dress", use_container_width=True)
            col2.metric("Fit score", f"{verdict.score}/100")
            col2.progress(verdict.score / 100)
            col2.subheader(verdict.verdict)
            col2.write(f"Detected: {verdict.analysis.summary()}")
            st.subheader("Why")
            for reason in verdict.reasons:
                st.write(f"- {reason}")
            with st.expander("Gemini's visual notes"):
                st.json(result["vision_notes"])

# ---------------------------------------------------------------------------
with tab_eval:
    st.header("Evaluation")
    st.caption("All metrics are written to reports/evaluation.md.")
    if st.button("Run retrieval evaluation (group split by image_name)"):
        with st.spinner("Evaluating..."):
            metrics = {"retrieval_group_split": retrieval_metrics(config.TRAINING_DATA_PATH, k=5)}
            path = write_report(metrics)
        st.json(metrics)
        st.success(f"Saved {path}")

    st.subheader("Body-shape evaluation")
    st.write("Upload a CSV with `shoulder_hip_ratio` and human-labelled `body_shape` columns.")
    labelled = st.file_uploader("Labelled body-shape CSV", type=["csv"], key="labels")
    if labelled and st.button("Evaluate body shape"):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(labelled.getvalue())
            tmp_path = Path(tmp.name)
        metrics = body_shape_metrics(tmp_path)
        tmp_path.unlink(missing_ok=True)
        write_report({"body_shape": metrics})
        st.json(metrics)

    st.subheader("Train/serve feature parity")
    if pipeline.profile:
        st.json(feature_parity(list(pipeline.profile.features)))
    else:
        st.info("Build a profile to check feature parity.")

# ---------------------------------------------------------------------------
with tab_label:
    st.header("Collect human fit ratings")
    st.caption(
        "Rate dresses against the current body profile. Each rating is appended "
        "to data/dress_ratings.csv, the exact schema used by the fit evaluation."
    )

    if "last_rating" in st.session_state:
        last = st.session_state["last_rating"]
        del st.session_state["last_rating"]
        delta = last["score"] - last["model_score"]
        st.success(
            f"Saved {last['score']}/100. The model predicted {last['model_score']}/100 "
            f"(difference {delta:+d})."
        )

    if pipeline.profile is None:
        st.info("Build a body profile in tab 1 first.")
    else:
        left, right = st.columns([2, 1])
        with left:
            source = st.radio(
                "Candidate source",
                ["Generated catalogue", "Uploaded dress photo"],
                horizontal=True,
            )
            candidate = None
            idx = 0

            if source == "Generated catalogue":
                if "label_candidates" not in st.session_state:
                    st.session_state["label_candidates"] = labeling.generate_candidates(n=200, seed=42)
                if "label_idx" not in st.session_state:
                    st.session_state["label_idx"] = 0
                candidates = st.session_state["label_candidates"]
                idx = st.session_state["label_idx"]
                if idx >= len(candidates):
                    st.success("Sample complete. Reset to start again.")
                    if st.button("Reset sample"):
                        st.session_state["label_idx"] = 0
                        st.rerun()
                else:
                    candidate = candidates[idx]
                    st.subheader(f"Candidate {idx + 1}/{len(candidates)}")
                    st.write(
                        f"**{candidate['length']} {candidate['silhouette']} dress** · "
                        f"colours: {candidate['colors'].replace('|', ', ')} · "
                        f"{candidate['occasion']} · {candidate['season']}"
                    )
            else:
                photo = st.file_uploader("Dress photo to rate", type=["jpg", "jpeg", "png"], key="rating_photo")
                if photo:
                    image = decode_image(photo.getvalue())
                    analysis = analyze_dress_cv(image)
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Dress to rate", use_container_width=True)
                    silhouette = st.selectbox(
                        "Silhouette",
                        labeling.SILHOUETTES,
                        index=labeling.SILHOUETTES.index(analysis.silhouette)
                        if analysis.silhouette in labeling.SILHOUETTES else 0,
                    )
                    length = st.selectbox(
                        "Length",
                        labeling.LENGTHS,
                        index=labeling.LENGTHS.index(analysis.length)
                        if analysis.length in labeling.LENGTHS else 1,
                    )
                    colors = st.text_input(
                        "Colours (pipe separated)",
                        value="|".join(analysis.colors) if analysis.colors else "blue",
                    )
                    candidate = labeling.candidate_from_analysis(
                        DressAnalysis(
                            silhouette=silhouette,
                            length=length,
                            colors=[c.strip() for c in colors.split("|") if c.strip()],
                        ),
                        occasion=occasion,
                        season=season,
                    )

            if candidate:
                score = st.slider(
                    "How well would this look on this body profile? (0 = poor, 100 = flawless)",
                    0, 100, 50, key=f"rating_{source}_{idx}",
                )
                submit_col, skip_col = st.columns(2)
                if submit_col.button("Submit rating", type="primary"):
                    labeling.append_rating(candidate, score)
                    model_score = score_fit(
                        pipeline.profile,
                        DressAnalysis(
                            category=candidate["category"],
                            silhouette=candidate["silhouette"],
                            length=candidate["length"],
                            colors=[c for c in candidate["colors"].split("|") if c],
                        ),
                        occasion=candidate["occasion"],
                        season=candidate["season"],
                        kb=knowledge,
                    ).score
                    st.session_state["last_rating"] = {"score": score, "model_score": model_score}
                    if source == "Generated catalogue":
                        st.session_state["label_idx"] = idx + 1
                    st.rerun()
                if source == "Generated catalogue" and skip_col.button("Skip"):
                    st.session_state["label_idx"] = idx + 1
                    st.rerun()

        with right:
            st.subheader("Progress")
            st.json(labeling.rating_summary())
            if st.button("Save profile for CLI evaluation"):
                st.success(f"Saved to {labeling.save_profile(pipeline.profile)}")
            ratings = labeling.load_ratings()
            if not ratings.empty:
                st.download_button(
                    "Download ratings CSV",
                    data=ratings.to_csv(index=False).encode("utf-8"),
                    file_name="dress_ratings.csv",
                )
                if st.button("Evaluate fit scorer with these ratings"):
                    metrics = dress_fit_metrics(labeling.RATING_PATH, pipeline.profile, knowledge)
                    write_report({"dress_fit": metrics})
                    st.json(metrics)
