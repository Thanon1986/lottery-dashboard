# Lottery Statistics Dashboard

## Project Overview

Lottery Statistics Dashboard is a local Streamlit app for exploring historical Thai lottery result data. It reads `data/lottery_history.csv`, generates summary workbooks, runs statistical scoring methods, and displays historical analysis dashboards.

This project is designed to run locally and to be moved to another computer without connecting to any company server.

## Features

- View lottery history from `data/lottery_history.csv`
- Add a new historical draw with validation and CSV backup
- Recalculate statistical summaries
- Frequency analysis for digits, last2, and 3-digit values
- Weighted recent trend scoring
- Hybrid scoring with `hybrid_score_v1`
- Hot and cold number analysis
- Digit position analysis
- Backtesting and model comparison dashboard
- Prediction history logging for saved statistical analyses
- Data quality report generation

## Folder Structure

```text
.
|- app.py
|- analyzer.py
|- predictor.py
|- requirements.txt
|- README.md
|- DEPLOY_CHECKLIST.md
|- data/
|  `- lottery_history.csv
|- output/
|  |- stat_summary.xlsx
|  |- prediction_history.csv
|  |- backtest_result.xlsx
|  |- backtest_detail.csv
|  `- data_quality_report.xlsx
|- backup/       (ignored by git)
|- logs/         (ignored by git)
`- temp/
```

## How to Run Locally

Install dependencies:

```powershell
py -m pip install -r requirements.txt
```

Run the dashboard:

```powershell
py -m streamlit run app.py
```

### Local Login Secrets

The dashboard is protected by a simple username/password gate using `st.secrets`.

Create a local file:

```text
.streamlit/secrets.toml
```

Use this format:

```toml
[auth]
username = "admin"
password = "your-real-password"
```

You can copy the template from:

```text
.streamlit/secrets.example.toml
```

Never put a real password in `app.py`, `README.md`, or any committed code file. The real `.streamlit/secrets.toml` file is ignored by git.

### Streamlit Cloud Login Secrets

On Streamlit Cloud:

1. Open the app settings.
2. Go to the Secrets section.
3. Add:

```toml
[auth]
username = "admin"
password = "your-real-password"
```

4. Save and restart the app.

Do not commit `.streamlit/secrets.toml` to GitHub. Commit only `.streamlit/secrets.example.toml`.

Optional command-line refresh:

```powershell
py analyzer.py
py predictor.py
```

## How to Move to Another Computer

1. Copy the project folder or clone the Git repository.
2. Confirm these files exist:
   - `app.py`
   - `analyzer.py`
   - `predictor.py`
   - `requirements.txt`
   - `data/lottery_history.csv`
   - `output/stat_summary.xlsx`
   - `output/prediction_history.csv`
   - `output/backtest_result.xlsx`
   - `output/backtest_detail.csv`
   - `output/data_quality_report.xlsx`
3. Install Python 3.10 or newer.
4. Run `py -m pip install -r requirements.txt`.
5. Run `py -m streamlit run app.py`.

The code uses relative paths such as `data/lottery_history.csv` and `output/stat_summary.xlsx`, so the project can be moved as one folder.

## How to Push to GitHub

Recommended first-time setup:

```powershell
git init
git status
git add app.py analyzer.py predictor.py requirements.txt README.md DEPLOY_CHECKLIST.md .gitignore data/lottery_history.csv output/stat_summary.xlsx output/prediction_history.csv output/backtest_result.xlsx output/backtest_detail.csv output/data_quality_report.xlsx
git commit -m "Prepare Lottery Statistics Dashboard"
git branch -M main
git remote add origin <your-private-repo-url>
git push -u origin main
```

Use a private repository unless you have reviewed privacy, access control, and data-sharing requirements.

## Safety Notes

- The dashboard reads and writes only local project files.
- Do not add secrets, passwords, API keys, or company-only files to Git.
- `backup/`, `logs/`, virtual environments, and secret files are ignored by `.gitignore`.
- Authentication uses `st.secrets`; never hardcode a real password in Python code.
- The app must not be connected to a company server unless a separate approved deployment plan exists.
- If deployed publicly, add authentication before publishing.

## Disclaimer

Historical statistical analysis only. This system does not predict or guarantee lottery outcomes.
