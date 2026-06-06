# Traffic Demand Prediction

Predicting normalized traffic demand for road segments at 15-minute intervals using a LightGBM + XGBoost + CatBoost ensemble.

**Validation R² : 0.9277**

---

## Project Structure

```
├── train.csv          # Training data (77,299 rows)
├── test.csv           # Test data (41,778 rows)
├── submission.csv     # Final predictions (generated after running the notebook)
└── trafficDemandPrediction.ipynb
```

---

## Problem Statement

Given road segment identifiers (geohash), timestamps, road properties, and weather data across two days, predict **traffic demand** — a normalized value between 0 and 1 — for each segment at 15-minute intervals.

**Evaluation metric:** R² (coefficient of determination)

---

## Pipeline Overview

| Step | Description |
|------|-------------|
| 1 | Geohash decoding → lat/lon coordinates |
| 2 | Static feature imputation (road type, lanes, etc.) |
| 3 | Temporal features (sin/cos time encoding) |
| 4 | Weather & temperature imputation |
| 5 | Day 48 historical statistics per geohash |
| 6 | 24-hour lag features with leakage guard |
| 7 | Categorical encoding |
| 8 | Time-aware train/validation split |
| 9 | Ensemble training & submission |

---

## Key Features

- **Sin/Cos time encoding** — captures the circular nature of time so the model understands 23:45 and 00:00 are only 15 minutes apart
- **24h lag features** — demand at a road segment yesterday is the strongest predictor of demand today; includes ±15 and ±30 minute offsets
- **Spatial lag** — average demand of the 3 nearest geographic neighbours at the same timestamp
- **Nearest-neighbour fallback** — geohashes absent from Day 48 borrow stats from their closest geohash by lat/lon distance
- **Leakage guard** — Day 48 lag values are set to NaN for Day 48 training rows to prevent target leakage

---

## Models

| Model | Validation R² |
|-------|--------------|
| LightGBM | 0.92120 |
| XGBoost | 0.92437 |
| CatBoost | 0.92588 |
| **Ensemble (avg)** | **0.92766** |

All three models use the same hyperparameters:
- `n_estimators = 1000`
- `learning_rate = 0.03`
- `subsample = 0.8`
- `colsample_bytree = 0.8`

Predictions are averaged and clipped to `[0.0, 1.0]`.

---

## Dataset Columns

| Column | Description |
|--------|-------------|
| `geohash` | Base-32 encoded road segment identifier |
| `day` | Day number (48 = train reference day, 49 = test day) |
| `timestamp` | Time in HH:MM format at 15-minute intervals |
| `RoadType` | Type of road |
| `NumberofLanes` | Number of lanes |
| `LargeVehicles` | Presence of large vehicles |
| `Landmarks` | Nearby landmarks |
| `Temperature` | Temperature at that timestamp |
| `Weather` | Weather condition |
| `demand` | Target — normalized traffic demand [0, 1] |

---

## How to Run

1. Open `traffic_pipeline.ipynb` in Google Colab
2. Upload `train.csv` and `test.csv` to the Colab root directory
3. Run all cells in order
4. `submission.csv` will be saved to the root directory

Note: The lag feature cells iterate row by row and will take a few minutes to complete — this is expected.

---

## Dependencies

```
pandas
numpy
scikit-learn
lightgbm
xgboost
catboost
```

Install via:
```bash
pip install lightgbm xgboost catboost -q
```
