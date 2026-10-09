# SYNOVA AI — ML Prediction API

**AI-powered delivery and installation risk prediction for retail operations.**

SYNOVA AI helps retail operations teams identify potentially delayed deliveries and installations before they become customer-facing problems.

This repository contains the machine learning API powering SYNOVA AI's experimental delivery and installation predictions.

## Overview

The API uses machine learning models trained on synthetic operational data to predict whether a delivery or installation may be delayed.

It is designed to support SYNOVA AI's core workflow:

* **Predict** — Estimate potential delivery and installation delays.
* **Explain** — Combine predictions with operational risk indicators.
* **Prevent** — Help operations teams prioritize proactive interventions.

> **Important:** The models are trained exclusively on synthetic data. Predictions are experimental and have not been validated against real-world operational data. Model scores must not be interpreted as calibrated real-world probabilities.

## Features

* Delivery delay prediction
* Installation delay prediction
* Separate prediction endpoints
* FastAPI-based REST API
* Pre-trained scikit-learn models
* Health-check endpoint
* Input validation and structured JSON responses

## Tech Stack

* Python
* FastAPI
* scikit-learn
* pandas
* NumPy
* joblib
* Uvicorn

## API Endpoints

| Method | Endpoint                | Description                     |
| ------ | ----------------------- | ------------------------------- |
| `GET`  | `/`                     | API information                 |
| `GET`  | `/health`               | Health check                    |
| `POST` | `/predict/delivery`     | Predict delivery delay risk     |
| `POST` | `/predict/installation` | Predict installation delay risk |

### Health Check

**Request**

```http
GET /health
```

Use this endpoint to check whether the API service is available.

### Delivery Prediction

**Request**

```http
POST /predict/delivery
Content-Type: application/json
```

Example request body:

```json
{
  "features": {
    "inventoryAvailableAtCreation": 1,
    "deliveryZone": "urban",
    "distanceKm": 25,
    "carrierId": "CARRIER-A",
    "dispatchStatusAtCutoff": "dispatched",
    "hoursToPromisedAtCutoff": 12,
    "installationRequired": 0,
    "installationSlotAvailableAtCutoff": 0,
    "installationSchedulingAtCutoff": "not_required"
  }
}
```

The example values are illustrative and must match the feature schema and accepted categories used by the deployed model.

### Installation Prediction

**Request**

```http
POST /predict/installation
Content-Type: application/json
```

Example request body:

```json
{
  "features": {
    "inventoryAvailableAtCreation": 1,
    "deliveryZone": "urban",
    "distanceKm": 25,
    "carrierId": "CARRIER-A",
    "dispatchStatusAtCutoff": "dispatched",
    "hoursToPromisedAtCutoff": 12,
    "installationRequired": 1,
    "installationSlotAvailableAtCutoff": 1,
    "installationSchedulingAtCutoff": "scheduled"
  }
}
```

Installation predictions require `installationRequired` to be enabled.

*Note: The request examples illustrate the expected nested `features` structure. Check the deployed API's response schema and feature validation before relying on specific response fields.*

## Model and Dataset

The models were trained using the versioned synthetic dataset:

`synova-synthetic-v1`

The dataset generator supports reproducible synthetic records using a fixed random seed.

### Label Definitions

* **Delivery delayed:** Delivery occurs more than 24 hours after the promised delivery date.
* **Installation delayed:** Installation occurs more than 72 hours after delivery.
* **Unknown:** The available outcome data is insufficient to assign a reliable label.
* **Not applicable:** Installation is not required for the order.

Prediction-time features are kept separate from outcome fields to reduce target leakage.

### Model Artifacts

The deployed service loads pre-trained model artifacts at startup:

* `delivery_model.joblib`
* `installation_model.joblib`
* `metadata.json`

These artifacts correspond to models trained on synthetic records. They should be evaluated on representative real-world data before being used for operational decisions.

## Local Development

### 1. Clone the Repository

```bash
git clone https://github.com/Umud-Guliyev/synova-ai-ml-api.git
cd synova-ai-ml-api
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the API

Use the application's actual FastAPI entry-point module:

```bash
uvicorn main:app --reload
```

If the application module has a different filename or location, adjust `main:app` accordingly.

### 5. Open the API Documentation

Once the server is running:

* Swagger UI: `http://127.0.0.1:8000/docs`
* ReDoc: `http://127.0.0.1:8000/redoc`
* Health check: `http://127.0.0.1:8000/health`

## Deployment

The API is deployed using Render.

**Production API:** https://synova-ai-ml-api.onrender.com

* Health check: https://synova-ai-ml-api.onrender.com/health
* Interactive API documentation: https://synova-ai-ml-api.onrender.com/docs

The service loads the model artifacts during startup. Availability and cold-start behavior depend on the deployment configuration and hosting plan.

## Evaluation and Limitations

The models were evaluated using synthetic data, including a chronological holdout split.

The chronological holdout results were:

| Metric            | Delivery Model | Installation Model |
| ----------------- | -------------: | -----------------: |
| Accuracy          |          0.889 |              0.644 |
| Balanced accuracy |          0.765 |              0.605 |
| Macro F1          |          0.777 |              0.606 |

These metrics describe performance on the synthetic evaluation data only. They do not establish real-world predictive performance.

Known limitations:

* Synthetic training data may not represent real retail operations.
* The models have not been validated on real operational outcomes.
* Prediction scores are not validated or calibrated real-world probabilities.
* Model performance may change across carriers, regions, products, and operational conditions.
* Human review should remain part of any operational decision.

**Recommended next step:** Evaluate the models on representative historical operational data, using a time-based holdout and metrics appropriate for delayed-order detection.

## Integration with SYNOVA AI

This API powers the experimental ML prediction feature in the SYNOVA AI application.

The application requests predictions explicitly when a user initiates them. The results complement the platform's risk analysis and intervention workflow; they do not independently trigger operational actions.

## License

Add the license appropriate for your project before distributing or reusing this repository.

---

**Built for the NeuroBridge Hackathon — AI Enterprise Solutions Track.**

SYNOVA AI — *Know the delay before it happens.*
