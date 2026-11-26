# AI Personal Stylist

A Streamlit app that turns **full-body photos + an occasion** into a grounded outfit recommendation, and can also judge whether a **specific dress** is likely to flatter the customer.

This is a rebuild of the original notebook project with the evaluation gap and the critical bugs fixed.

## What it does

1. **Body profile from one or more photos** — MediaPipe Pose extracts 33 landmarks; a single shared feature extractor computes 43 anthropometric ratios. Multiple photos are fused with a weighted median (fitted-clothes photos count 2×), and the person mask gives an extra shoulder/hip silhouette cue.
2. **Body-type estimate** — a rule-based classifier returns a shape, an explainable heuristic confidence, cross-photo agreement and stability (so you can see when the user's photos disagree).
3. **Similar profiles + fashion rules** — the body embedding is matched against DeepFashion2-derived profiles with FAISS; the retrieved garment categories select styling rules from a 250-row knowledge base.
4. **Gemini recommendation** — the profile, similar items and rules form a grounded prompt; Gemini returns a structured, occasion-aware outfit plan.
5. **Dress check** — upload a dress photo. OpenCV estimates silhouette (bodycon / straight / A-line / fit-and-flare), length, colours and pattern; Gemini Vision corrects/refines it; an explainable 0–100 fit score compares it with the customer's profile.
6. **Rating collection** — Tab 5 generates reproducible dress candidates (or uses an uploaded dress photo) and records your 0–100 judgement against the current profile. Ratings land in `data/dress_ratings.csv`, ready for the fit evaluation.

> The app **requires** a `GOOGLE_API_KEY` by design. It does not silently fall back to canned text.

## Architecture

```
photos + occasion
      │
      ▼
MediaPipe Pose ──► landmarks ──► src/features.py (ONE extractor)
      │                              │
      │                              ├─► src/profile.py  (multi-photo fusion)
      │                              └─► src/body_shape.py (shape + confidence)
      │                                        │
      │                    ┌───────────────────┴───────────────────┐
      │                    ▼                                       ▼
      │           src/retrieval.py (FAISS)              src/dress.py (CV + fit score)
      │                    │                                       │
      │                    ▼                                       ▼
      │           src/knowledge.py (rules)              Gemini Vision refinement
      │                    │                                       │
      └────────────────────┴──────────────► src/llm.py ◄────────────┘
                                                  │
                                                  ▼
                                   Streamlit UI (app.py)
```

## Quick start

```bash
cd ai_stylist
python -m venv .venv
.venv\Scripts\activate          # Windows:  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
copy .env.example .env          # then paste your key from Google AI Studio
python scripts/build_index.py   # one-time: fit scaler + PCA + FAISS
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

> **Dependency note:** MediaPipe 1.x removed the `solutions` API this app uses, and protobuf 5 conflicts with MediaPipe 0.10.x. `requirements.txt` therefore pins `mediapipe<1` and `streamlit<1.38` (which supports protobuf 4). If you already have newer versions, use a fresh virtual environment.

**Environment variables** (in `.env`):

| Variable | Required | Default |
|---|---|---|
| `GOOGLE_API_KEY` | yes | — |
| `GEMINI_MODEL` | no | `gemini-3.6-flash` |

Use a currently documented endpoint; see [Google's model list](https://ai.google.dev/gemini-api/docs/models).

## Using the app

- **Tab 1 – Body profile:** upload 1–3 full-body photos, mark which show fitted clothes, click *Analyse photos*, then *Build profile*. The app shows detected landmarks, shape, confidence, agreement and measurement stability.
- **Tab 2 – Outfit:** choose occasion and season in the sidebar, then *Generate recommendation*. Expand the sections to inspect the exact FAISS neighbours and knowledge-base rows used as evidence.
- **Tab 3 – Dress check:** upload a dress photo (plain background works best) and click *Evaluate dress*. You get a 0–100 score, a verdict, per-factor reasons and Gemini's visual notes.
- **Tab 4 – Evaluation:** run retrieval metrics, upload a labelled body-shape CSV, and check train/serve feature parity. Results are written to `reports/evaluation.md`.
- **Tab 5 – Collect ratings:** pick *Generated catalogue* for a deterministic stream of dress descriptions, or *Uploaded dress photo* to rate a real garment. Set a 0–100 slider, submit, and the model's prediction is revealed afterwards so it cannot bias your rating. Use the side panel to save the profile, download the CSV, or compute Spearman ρ / MAE immediately.

## Project structure

```
ai_stylist/
├── app.py                     # Streamlit UI
├── config.py                  # paths, thresholds, model name
├── requirements.txt
├── .env.example
├── data/
│   ├── fashion_knowledge_base.csv      # 250 styling rules
│   ├── body_features_with_metadata.csv # 3,345 people / 5,820 garment rows
│   └── dress_ratings.csv               # generated by Tab 5 (human labels)
├── models/                    # generated by scripts/build_index.py
├── reports/                   # generated evaluation.md
├── scripts/
│   ├── build_index.py
│   └── evaluate.py
├── src/
│   ├── features.py            # single source of truth (43 features)
│   ├── pose.py                # MediaPipe + silhouette cue
│   ├── profile.py             # multi-photo weighted-median fusion
│   ├── body_shape.py          # heuristic shape + confidence
│   ├── retrieval.py           # FAISS, dedup, context builder
│   ├── knowledge.py           # category mapping + rule ranking
│   ├── dress.py               # CV analysis + explainable fit score
│   ├── llm.py                 # Gemini (new SDK), required key
│   ├── pipeline.py            # orchestration
│   ├── labeling.py            # rating catalogue + CSV writer
│   └── evaluation.py          # metrics
└── tests/test_core.py
```

## Evaluation

Run `python scripts/evaluate.py` or use tab 4. After collecting ratings in tab 5:

```bash
python scripts/evaluate.py --dress-ratings data/dress_ratings.csv --profile-pickle models/user_profile.joblib
```

| Layer | Metric | Status / note |
|---|---|---|
| Feature engineering | Train/serve column parity | Automated check in tab 4 and tests. This catches the original `pose_score` skew. |
| Retrieval | Recall@5, MRR@5, nDCG@5, unique identities@5, duplicate rate, self-match rate | Group split by `image_name`; relevance proxy = same body-shape label. See caveat below. |
| Body shape | Accuracy, **macro-F1**, per-class recall, confusion matrix, majority baseline | Requires a labelled CSV. The bundled data is 99.3% one class, so macro-F1 is the metric that matters. |
| Dress fit | Spearman ρ and MAE vs human 0–100 ratings | Tab 5 writes `human_score`, `category`, `silhouette`, `length`, `colors`, `occasion`, `season` automatically. Aim for ~50 ratings before trusting ρ. |
| Generation | Faithfulness, completeness, latency, tokens | Add a judged sample of 30 Gemini outputs; the harness is intentionally extensible. |

The thresholds in `src/body_shape.py` (1.58 / 1.80) are **calibrated to this dataset's distribution** (median ratio ≈ 1.70, biased high by partially occluded hip landmarks). They avoid the original project's 99.3%-one-class collapse, but they are not ground truth. Replace them with a calibrated model once labels exist.

## Limitations

- Body measurements are **relative to pose estimates**, not centimetres. Gemini is instructed to phrase the shape as an estimate.
- MediaPipe can fail on cropped, occluded or unusual poses; low-quality photos are rejected rather than guessed.
- The dress fit score is an **explainable heuristic + LLM judgement**, not a trained per-garment fit model. A white/plain background gives the best CV silhouette.
- The bundled FAISS data has ~1.7 garment rows per person; results are deduplicated by image before being shown.
- No offline fallback: without a valid API key the app stops with an error.

## Attribution

The bundled CSVs are derived from DeepFashion2 annotations and the original project's fashion knowledge base. Please cite DeepFashion2 if you publish results based on them.
