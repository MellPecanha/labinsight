# LabInsight

LabInsight is an experimental platform for exploring statistical patterns in
public laboratory data. It combines data preparation, unsupervised machine
learning, a REST API, and an explanatory interface in a reproducible workflow.

The project uses the public NHANES `BIOPRO_L` file from the 2021–2023 cycle as
its initial reference dataset. It is designed for technical exploration and
learning. It is **not** a medical product and does not provide diagnoses,
clinical assessments, triage, or treatment recommendations.

## What it does

Given a panel of up to 18 laboratory variables, LabInsight:

1. applies the same preprocessing used during training;
2. calculates an Isolation Forest anomaly score;
3. marks whether the score is above an experimentally configured statistical
   threshold;
4. reports how many values were observed and how many were missing; and
5. presents the result together with its methodological limitations.

A high score means that a combination of values is statistically less common in
the experiment's reference data. It does not identify a disease or establish a
clinical conclusion.

## Architecture

```text
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
```

The Streamlit application communicates with the local API. The API loads the
model artifact locally. No submitted values are sent to an external LLM or API.

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

## Project structure

```text
app/                    FastAPI application, schemas, and model service
ml/preprocessing/       NHANES loading and feature selection
ml/training/            Isolation Forest training and experiment runner
ml/notebooks/           Exploratory analysis notebook
data/raw/               Local public source files, ignored by Git
data/processed/         Local model artifacts, ignored by Git
tests/                  ML, API, and interface helper tests
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
- Missing values are median-imputed. Completeness is returned with every result
  because incomplete panels can affect the score.
- The model does not use NHANES survey weights in the anomaly calculation.
  Survey weighting is intended for population estimation and is not assumed to
  be the correct objective for this individual-level statistical experiment.
- A statistical anomaly can result from many factors, including data quality,
  sampling differences, or a rare combination of otherwise valid measurements.
- Use public or synthetic values only. Do not submit personal or identifiable
  health information.

## Next steps

- Add a second detector as a clearly labelled sensitivity comparison.
- Add RAG over public NHANES methodology documentation, keeping retrieved
  explanations separate from the model score.
- Containerize the API and interface for a reproducible local deployment.
