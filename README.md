# AeroMaintain

AeroMaintain is an end-to-end machine learning project for predicting the Remaining Useful Life (RUL) of aircraft engines from multivariate time-series sensor data.

The project is based on the NASA C-MAPSS turbofan engine degradation dataset and goes beyond model training by providing uncertainty estimation, an inference API, an interactive dashboard, automated tests, continuous integration, and a containerized deployment architecture.

## Overview

Predictive maintenance aims to estimate when a component is likely to fail before the failure actually occurs.

AeroMaintain uses historical operating conditions and sensor measurements from aircraft engines to estimate their Remaining Useful Life.

Instead of returning only a single prediction such as:

> The engine has 25 cycles remaining.

the application also estimates prediction quantiles:

* q10: lower RUL estimate
* q50: median RUL estimate
* q90: upper RUL estimate

This provides a more informative representation of prediction uncertainty.

## Main Features

* Remaining Useful Life prediction
* Temporal feature engineering from sensor histories
* Comparison of multiple regression algorithms
* Model selection using a validation dataset
* Quantile regression with q10, q50 and q90 predictions
* Quantile calibration and coverage evaluation
* Prediction bias analysis
* FastAPI inference API
* Pydantic request validation
* Streamlit dashboard
* Automated tests with pytest
* Continuous Integration with GitHub Actions
* Dockerized backend and frontend
* Docker Compose orchestration
* API healthcheck and service dependency management

## Dataset

AeroMaintain uses the NASA C-MAPSS turbofan engine degradation dataset.

Each engine is represented by a sequence of operating cycles containing:

* engine identifier
* cycle number
* 3 operating condition parameters
* 21 sensor measurements

The different C-MAPSS subsets are identified as:

* FD001
* FD002
* FD003
* FD004

The objective is to predict the Remaining Useful Life of an engine from its operating history.

## Machine Learning Pipeline

The machine learning workflow follows a train, validation and test strategy.

```text
Training data
     |
     v
Temporal feature engineering
     |
     v
Train candidate models
     |
     v
Validation set
     |
     v
Select best model
     |
     v
Retrain on train + validation
     |
     v
Final evaluation on test set
```

The test set is only used after model selection in order to estimate final generalization performance.

## Temporal Feature Engineering

The raw C-MAPSS sensor measurements are enriched with temporal information.

For each sensor, AeroMaintain generates features including:

* difference from the previous cycle
* rolling mean over 5 cycles
* rolling standard deviation over 5 cycles
* rolling mean over 20 cycles
* rolling standard deviation over 20 cycles
* approximate trend over 20 cycles

Features are computed independently for each engine and dataset.

Only the current cycle and previous cycles are used when creating temporal features, preventing information from future cycles from leaking into predictions.

## Candidate Models

Several regression models can be compared during training, including:

* Linear Regression
* Ridge Regression
* Lasso Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* Histogram Gradient Boosting
* Support Vector Regression
* K-Nearest Neighbors
* Multi-Layer Perceptron

The best model is selected using validation RMSE.

After selection, the winning model is retrained using the combined training and validation datasets before being evaluated on the final test set.

## Evaluation Metrics

Point predictions are evaluated using:

### RMSE

Root Mean Squared Error gives more importance to large prediction errors.

### MAE

Mean Absolute Error measures the average absolute difference between predicted and actual RUL.

### R²

The coefficient of determination measures how much of the variability in RUL is explained by the model.

### Dummy baseline

The final model is also compared against a `DummyRegressor` baseline to verify that the machine learning model provides meaningful predictive value.

## Prediction Bias

AeroMaintain also evaluates whether the model tends to overestimate or underestimate Remaining Useful Life.

The analysis includes:

* mean signed error
* median error
* overestimation rate
* underestimation rate
* average overestimation
* average underestimation

This distinction is particularly important in predictive maintenance because overestimating the remaining lifetime of a degraded component can be more problematic than underestimating it.

## Quantile Regression

In addition to the main RUL model, AeroMaintain trains three Gradient Boosting quantile regressors:

```text
q10 -> 10th percentile
q50 -> median
q90 -> 90th percentile
```

A prediction can therefore be represented as:

```text
q10 < q50 < q90
```

For example:

```text
q10 = 18 cycles
q50 = 27 cycles
q90 = 41 cycles
```

The median estimate is 27 remaining cycles, while the q10-q90 interval indicates the range represented by the lower and upper predicted quantiles.

## Quantile Evaluation

The quantile models are evaluated using several diagnostics.

## Results

The final selected model was evaluated on a held-out test set that was not used during model selection.

### RUL Prediction

| Metric | Test result |
|---|---:|
| RMSE | 54.97 cycles |
| MAE | 40.32 cycles |
| R² | 0.519 |

The model explains approximately 52% of the variance in Remaining Useful Life on the final test set.

The MAE indicates an average absolute prediction error of approximately 40 cycles.

### Quantile Prediction

| Metric | Target | Test result |
|---|---:|---:|
| q10-q90 coverage | 80% | 79.08% |
| Mean interval width | - | 139.11 cycles |
| q10 calibration | 10% | 7.95% |
| q50 calibration | 50% | 42.09% |
| q90 calibration | 90% | 87.02% |
| Quantile crossing rate | 0% | 0.10% |

The q10-q90 interval reaches a coverage of 79.08%, very close to its theoretical 80% target.

Quantile crossing is rare, occurring in approximately 0.10% of predictions.

The median quantile q50 has a mean signed error of:

```text
-20.97 cycles

### Coverage

Coverage measures the proportion of actual RUL values located between q10 and q90.

Since the interval spans the 10th to the 90th percentile, its theoretical target coverage is approximately 80%.

### Interval Width

The average value of:

```text
q90 - q10
```

measures the width of the predicted uncertainty interval.

Narrow intervals are more informative, provided that they remain sufficiently well calibrated.

### Pinball Loss

Pinball loss evaluates the quality of individual quantile predictions while penalizing errors differently depending on the target quantile.

It is calculated independently for q10, q50 and q90.

### Quantile Calibration

The project checks whether approximately:

```text
10% of true values are below q10
50% of true values are below q50
90% of true values are below q90
```

### Quantile Crossing

The pipeline also measures invalid predictions where:

```text
q10 > q50
```

or:

```text
q50 > q90
```

### Calibration by Wear Level

Quantile performance can also be analyzed for different RUL ranges:

```text
RUL <= 10
10 < RUL <= 20
20 < RUL <= 30
30 < RUL <= 60
RUL > 60
```

This makes it possible to determine whether uncertainty estimation behaves differently for healthy and highly degraded engines.

## Architecture

AeroMaintain separates the user interface, inference API and machine learning logic.

```text
                         User
                          |
                          v
                 Streamlit Dashboard
                      port 8501
                          |
                          | HTTP
                          v
                    FastAPI API
                      port 8000
                          |
                          v
                 Request validation
                       Pydantic
                          |
                          v
                DataFrame conversion
                          |
                          v
              Temporal feature engineering
                          |
                          v
              Trained scikit-learn models
                    /             \
                   /               \
                  v                 v
            RUL prediction    Quantile models
                              q10 q50 q90
                   \               /
                    \             /
                          v
                     JSON response
                          |
                          v
                 Streamlit Dashboard
```

The Streamlit application communicates with FastAPI exclusively through HTTP.

This separation allows the frontend and backend to be deployed independently.

## Project Structure

```text
AirMaintain-ML-Aircraft-engine-prediction/
|
|-- .github/
|   `-- workflows/
|       `-- ci.yml
|
|-- dashboard/
|   |-- app.py
|   |-- Dockerfile
|   `-- requirements.txt
|
|-- dataset/
|   |-- train/
|   |-- X_test/
|   |-- y_test/
|   |-- entry/
|   `-- results/
|       |-- models/
|       `-- predictions/
|
|-- scripts/
|   `-- train.py
|
|-- src/
|   `-- aeromaintain/
|       |-- api/
|       |   |-- main.py
|       |   |-- schemas.py
|       |   `-- converters.py
|       |
|       |-- data/
|       |-- features/
|       |   `-- temporal.py
|       |
|       |-- inference/
|       |   `-- predictor.py
|       |
|       |-- models/
|       `-- config.py
|
|-- tests/
|   |-- test_api.py
|   |-- test_converters.py
|   |-- test_predictor.py
|   |-- test_quantiles.py
|   |-- test_split.py
|   `-- test_temporal.py
|
|-- Dockerfile
|-- compose.yaml
|-- pyproject.toml
`-- README.md
```

## FastAPI Backend

The inference backend is implemented with FastAPI.

Main endpoints include:

```text
GET  /health
GET  /version
GET  /model/info
POST /predict
```

Interactive API documentation is automatically available through Swagger UI.

When running locally with Docker Compose:

```text
http://localhost:8000/docs
```

### Prediction Request

The `/predict` endpoint receives:

* dataset identifier
* engine identifier
* complete engine history
* operating parameters for each cycle
* 21 sensor values for each cycle

The backend then:

1. validates the request using Pydantic
2. converts the request into a pandas DataFrame
3. generates temporal features
4. selects the latest engine cycle
5. runs the trained RUL model
6. runs the q10, q50 and q90 models
7. returns the predictions as JSON

## Streamlit Dashboard

The Streamlit dashboard provides a graphical interface for interacting with the inference API.

It allows the user to:

* check API availability
* inspect model information
* upload a C-MAPSS engine history file
* preview the uploaded data
* select an engine
* select the corresponding FD dataset
* request a RUL prediction
* visualize q10, q50 and q90 estimates

When running with Docker Compose:

```text
http://localhost:8501
```

## Docker Architecture

The application contains two Docker services.

```text
docker compose
|
|-- api
|   |-- FastAPI
|   |-- machine learning package
|   `-- trained models
|
`-- dashboard
    |-- Streamlit
    |-- pandas
    `-- requests
```

Inside the Docker Compose network, the dashboard communicates with the backend using:

```text
http://api:8000
```

From the host machine, the applications are available through:

```text
FastAPI:    http://localhost:8000
Streamlit:  http://localhost:8501
```

The API also includes a Docker healthcheck.

The dashboard is configured to depend on the API becoming healthy before starting.

## Installation

### Clone the repository

```bash
git clone "https://github.com/AntoninR-cmd/AirMaintain-ML-Aircraft-engine-prediction.git"
cd AirMaintain-ML-Aircraft-engine-prediction
```

### Create a Python environment

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the project and development dependencies:

```powershell
python -m pip install -e ".[dev]"
```

## Model Training

The C-MAPSS dataset must be available in the expected dataset directories before training.

The training pipeline generates:

* validation results
* final test results
* trained RUL model
* q10 model
* q50 model
* q90 model
* prediction files
* quantile evaluation results

The serialized models are stored under:

```text
dataset/results/models/
```

Model artifacts are intentionally excluded from Git version control.

## Running with Docker Compose

Once the trained model files are available, the complete application can be started with:

```bash
docker compose up -d --build
```

Check the services with:

```bash
docker compose ps
```

A healthy deployment should contain both:

```text
api
dashboard
```

with the API reporting a healthy status.

The dashboard can then be opened at:

```text
http://localhost:8501
```

The FastAPI documentation is available at:

```text
http://localhost:8000/docs
```

To stop the application:

```bash
docker compose down
```

## Running Tests

The project uses pytest.

Run the complete test suite with:

```bash
python -m pytest tests -v
```

The tests cover several important parts of the system, including:

* temporal feature calculations
* separation of engines between validation and test sets
* quantile coverage
* quantile crossing
* inference logic
* request conversion
* API endpoints
* invalid API requests

## Continuous Integration

GitHub Actions automatically executes the test suite when changes are pushed to the repository.

The CI workflow:

```text
Checkout repository
        |
        v
Setup Python
        |
        v
Install project
        |
        v
Install development dependencies
        |
        v
Run pytest
```

This helps ensure that changes do not silently break existing functionality.

## Reproducibility

The project uses fixed random seeds where applicable to improve reproducibility.

Examples include:

```text
random_state = 42
```

for dataset splitting and compatible machine learning models.

Dependencies are managed through `pyproject.toml`.

Docker provides an additional reproducible runtime environment for inference.

## Generated Artifacts

Training and evaluation produce several local artifacts, including:

```text
dataset/results/models/
dataset/results/predictions/
dataset/results/resultats_validation.csv
dataset/results/resultats_quantiles_test.csv
```

Large trained model files and generated prediction files are excluded from the Git repository.

## Current Scope

AeroMaintain focuses on the machine learning and software engineering pipeline required to transform sensor histories into deployable RUL predictions.

The current version includes:

```text
Data ingestion
     |
Feature engineering
     |
Model training
     |
Model validation
     |
Final testing
     |
Uncertainty estimation
     |
Model serialization
     |
Inference API
     |
Web dashboard
     |
Automated testing
     |
Continuous integration
     |
Containerized deployment
```

## Possible Future Improvements

Future versions could explore:

* hyperparameter optimization
* more advanced time-series models
* recurrent neural networks or Transformers
* model monitoring
* prediction drift detection
* maintenance decision optimization
* cloud deployment
* experiment tracking
* richer uncertainty calibration methods

These extensions are intentionally outside the scope of the current version.

## Technologies

The project uses:

```text
Python
pandas
NumPy
scikit-learn
joblib
FastAPI
Pydantic
Streamlit
pytest
Docker
Docker Compose
GitHub Actions
```

## Author

Antonin RIVRON
Developed as an end-to-end machine learning engineering project around predictive maintenance and aircraft engine Remaining Useful Life estimation.
