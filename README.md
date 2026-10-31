# LabInsight

LabInsight is an educational platform for structured reading of laboratory
reports and exploration of public laboratory data. It combines transparent
interval comparison, RAG, LLM integration, unsupervised machine learning, a
REST API, and an explanatory interface in a reproducible workflow.

The project uses the public NHANES `BIOPRO_L` file from the 2021–2023 cycle as
its initial reference dataset. It is designed for technical exploration and
learning. It is **not** a medical product and does not provide diagnoses,
clinical assessments, triage, or treatment recommendations.

## What it does

Given a synthetic or public laboratory report, LabInsight:

1. compares each value with the reference interval supplied by that same report;
2. labels the result as below, within, or above that supplied interval;
3. explains the limit of this purely mathematical comparison;
4. offers grounded questions about methodology and source data through RAG; and
5. keeps Isolation Forest as an advanced workflow for public NHANES exploration.

The interval comparison does not identify a disease or establish a clinical
conclusion. An Isolation Forest score, when used in the advanced workflow,
means only that a combination of values is statistically less common in the
experiment's public reference data.

The interface displays readable laboratory names and SI units. NHANES variable
codes remain visible only as technical references. It also offers a complete,
de-identified public example panel so the workflow can be demonstrated without
inventing values or entering personal information.

## Architecture

```text
Synthetic or public report
        |
        v
Deterministic interval comparison
        |
        v
Below / within / above supplied interval

NHANES BIOPRO_L (.xpt)
        |
        v
Feature selection and preprocessing
        |
        v
Isolation Forest artifact (.joblib)
        |
        v
FastAPI  <----------------  Streamlit interface
POST /v1/analyses                 local user input
        |
        v
Statistical score + completeness + limitations

Curated local documents --> TF-IDF retrieval --> optional Gemini explanation
```

The Streamlit application communicates with the local API. The API loads the
model artifact locally. Laboratory-panel values are never sent to an external
LLM or API.

## Dataset and scope

- **Primary dataset:** NHANES 2021–2023, `BIOPRO_L` public laboratory profile.
- **Exploratory demographic file:** `DEMO_L`, used only to inspect missingness
  patterns by age group during exploration.
- **Identifier:** `SEQN` is not a model feature and is never accepted by the
  API.
- **Feature panel:** 18 selected laboratory variables, defined in
  [`ml/preprocessing/biopro.py`](ml/preprocessing/biopro.py).

The original `.xpt` files and trained artifacts are excluded from Git by
design. Store them locally under `data/raw/` and `data/processed/`.

## Methodology

### Educational report reader

The report reader does not contain a clinical table of “normal” values. For
each item, it receives the result, unit, lower bound, and upper bound shown on
the same report, then performs a transparent numeric comparison:

- result < lower bound: below the supplied interval;
- lower bound ≤ result ≤ upper bound: within the supplied interval; and
- result > upper bound: above the supplied interval.

This is intentionally separate from the RAG and LLM layers. The comparison is
deterministic and auditable; Gemini does not receive the report values. The
included report is fictional and exists only to demonstrate the workflow.

### Preprocessing

- The known SAS XPORT zero-reading artifact
  (`5.397605346934028e-79`) is converted to `0.0`.
- Participants with no observed value across all 18 selected features are
  excluded from modelling.
- Partially observed panels are retained.
- Median imputation and `RobustScaler` are fitted **only on training data** and
  then applied to the holdout set or API input.
- Values are not removed simply because they are extreme. In this experiment,
  unusual values are precisely what the detector is intended to surface for
  statistical review.

### Isolation Forest

Isolation Forest builds random decision trees. Observations isolated after
fewer random splits receive a higher anomaly score. The model uses 300 trees
and a fixed random seed for reproducibility.

The operational threshold is the 98th percentile of training anomaly scores.
This selects roughly the top 2% of training records for statistical review.
The 2% setting represents review capacity in the experiment, **not** disease
prevalence.

### Unsupervised evaluation

There are no labelled clinical anomalies in this experiment. Therefore,
accuracy, precision, recall, ROC-AUC, and similar supervised metrics are not
reported. The 80/20 random holdout is used only to inspect score stability on
unseen records.

Current experiment result:

| Holdout records | Flagged records | Flagged fraction |
| ---: | ---: | ---: |
| 1,274 | 28 | 2.20% |

The missingness sensitivity check found that 27 of the 28 flagged holdout
records had all 18 selected features present. This does not validate a clinical
meaning; it only indicates that missing values were not the main driver of the
observed flags in this split.

## Grounded explanations (RAG + Gemini)

LabInsight includes a small, inspectable Retrieval-Augmented Generation flow
for questions about its methodology and source data:

1. The local TF-IDF retriever searches curated Markdown documents in
   `rag/knowledge_base/`.
2. The API returns no answer when no source is relevant.
3. When configured, Gemini receives only the user's methodological question and
   the retrieved context, then produces a Portuguese explanation constrained by
   a non-clinical system instruction.
4. The response returns the titles and URLs of the retrieved sources.

The RAG endpoint does not accept a laboratory panel. Do not include personal or
identifiable health information in questions sent to a third-party LLM.

To enable Gemini locally, copy the example file and add your own API key:

```bash
cp .env.example .env
```

Then set `GEMINI_API_KEY` in `.env`, restart the API, and use `POST
/v1/explanations` or the “Pergunte sobre o experimento” section in Streamlit.
The Gemini REST integration follows the official
[generateContent documentation](https://ai.google.dev/api/generate-content).

## Project structure

```text
app/                    FastAPI application, schemas, and model service
ml/preprocessing/       NHANES loading and feature selection
ml/training/            Isolation Forest training and experiment runner
ml/notebooks/           Exploratory analysis notebook
data/raw/               Local public source files, ignored by Git
data/processed/         Local model artifacts, ignored by Git
tests/                  ML, API, and interface helper tests
rag/                    Curated knowledge base and local retrieval logic
streamlit_app.py        Local Streamlit interface
```

## Run locally

### Requirements

- Python 3.13+
- A virtual environment with the packages in `requirements-dev.txt`
- `data/raw/BIOPRO_L.xpt` to train the model

Install dependencies if needed:

```bash
./.venv/bin/pip install -r requirements-dev.txt
```

### Train the model

Place the public `BIOPRO_L.xpt` file in `data/raw/`, then run:

```bash
./.venv/bin/python -m ml.training.run_isolation_forest \
  --input data/raw/BIOPRO_L.xpt \
  --output data/processed/isolation_forest.joblib
```

The command prints a holdout stability summary and writes the local model
artifact.

### Start the API

```bash
./.venv/bin/uvicorn app.main:app --reload
```

API documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

Example request:

```bash
curl -X POST http://127.0.0.1:8000/v1/analyses \
  -H "Content-Type: application/json" \
  -d '{"LBXSATSI": 12.5, "LBDSALSI": 42.0}'
```

### Start the interface

In a second terminal, with the API running:

```bash
./.venv/bin/streamlit run streamlit_app.py
```

Open the URL printed by Streamlit, normally
[http://localhost:8501](http://localhost:8501).

## Run with Docker

Docker Compose starts the API and the interface as separate services. The model
artifact remains local and is mounted read-only into the API container.

First ensure `data/processed/isolation_forest.joblib` exists, then run:

```bash
docker compose up --build
```

The API is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
and the interface at [http://localhost:8501](http://localhost:8501). Stop both
services with `Ctrl+C`, then run `docker compose down` if you want to remove
the containers and network.

## Testing

```bash
./.venv/bin/python -m pytest -q
```

The test suite covers the core Isolation Forest workflow, API validation and
responses, and the interface's input parsing helper.

## Limitations and responsible use

- This model is trained on a selected set of public NHANES measurements. Its
  output cannot be generalized as a clinical reference range.
- Reference intervals can vary by laboratory, assay, unit, and characteristics
  of the population. LabInsight therefore compares only with the interval
  supplied by the same report and does not maintain its own clinical ranges.
- Missing values are median-imputed. Completeness is returned with every result
  because incomplete panels can affect the score.
- The model does not use NHANES survey weights in the anomaly calculation.
  Survey weighting is intended for population estimation and is not assumed to
  be the correct objective for this individual-level statistical experiment.
- A statistical anomaly can result from many factors, including data quality,
  sampling differences, or a rare combination of otherwise valid measurements.
- Use public or synthetic values only. Do not submit personal or identifiable
  health information.
- Gemini explanations are optional and source-limited. They do not replace the
  retrieved documentation and must not be used for health decisions.

## Next steps

- Add a second detector as a clearly labelled sensitivity comparison.
- Containerize the API and interface for a reproducible local deployment.
