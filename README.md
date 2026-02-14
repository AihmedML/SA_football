# Saudi Pro League Winner Prediction 2025/26

A data-driven machine learning project that predicts the Saudi Pro League (Roshn Saudi League) champion using real-time stats from the FotMob API.

## How It Works

1. **Data Collection** — Scrapes 27 team stat categories (xG, possession, shots, tackles, etc.), current standings, fixtures, and up to 5 historical seasons from the FotMob API.
2. **Feature Engineering** — Builds 50+ features per team including per-match rates, xG metrics, recent form, and historical pedigree.
3. **ML Models** — Trains and evaluates 4 models (Random Forest, Gradient Boosting, XGBoost, Linear Regression) using Leave-One-Out cross-validation (ideal for 18-team datasets).
4. **Monte Carlo Simulation** — Runs 10,000 match-by-match simulations of the remaining season to produce championship probabilities.

## Setup

```bash
pip install -r requirements.txt
```

### Dependencies

- `requests` — API calls
- `pandas`, `numpy` — data processing
- `scikit-learn`, `xgboost` — ML models
- `matplotlib`, `seaborn` — visualizations
- `jupyter` — notebook runtime

## Usage

Open and run the notebook top to bottom:

```bash
jupyter notebook saudi_league_prediction.ipynb
```

Or run it in VS Code with the Jupyter extension.

## Project Structure

```
SA_football/
├── saudi_league_prediction.ipynb   # Main notebook
├── write_notebook.py               # Notebook generator script
├── requirements.txt                # Python dependencies
└── README.md
```

## Data Source

All data is fetched live from the [FotMob](https://www.fotmob.com/) public API — no static datasets or manual entry. The notebook handles:

- Current season standings and stats (18 teams, 27 stat categories)
- Match fixtures and results (for form calculation)
- Historical season standings (weighted low to avoid bias toward past performance)

## Notes

- Historical data is intentionally given low weight — predictions are driven by current season performance.
- The polite `time.sleep(1.5)` between API calls keeps requests respectful. Full data collection takes ~1-2 minutes.
- Leave-One-Out CV is used instead of train/test split because the dataset is only 18 rows (one per team).
