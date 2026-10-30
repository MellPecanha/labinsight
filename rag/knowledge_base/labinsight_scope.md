# LabInsight experimental scope

Source: local project documentation

LabInsight is an experimental software platform for exploring statistical
patterns in public laboratory data. It uses an Isolation Forest model trained
on selected variables from the NHANES BIOPRO_L dataset.

The anomaly score expresses how statistically uncommon a combination of values
is relative to the experiment's reference data. A high score is not a diagnosis,
does not identify a disease, and cannot support a treatment recommendation.

The model accepts partially observed panels. Missing values are median-imputed
using values learned during training, and the response reports how many values
were observed and missing. Incomplete panels can affect the score.

The 2% threshold is an operational statistical review threshold derived from
training scores. It is not a prevalence estimate and does not define clinical
normality or abnormality.
