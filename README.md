**Advanced Trading Analytics Dashboard**
Welcome to the Advanced Trading Analytics Dashboard project! This repository contains a powerful Streamlit dashboard (dashboard.py) for analyzing trading performance and a Jupyter notebook (trader_sentiment_analysis.ipynb) for exploring trader sentiment using the Fear and Greed Index.
Jump to Installation | Jump to Usage | Jump to Features

Project Overview
This project provides tools to analyze cryptocurrency trading performance, predict trade outcomes, and study market sentiment impacts. The dashboard offers interactive visualizations and predictive analytics, while the notebook dives into sentiment-driven trade success analysis.
Key Components

Streamlit Dashboard (dashboard.py): Visualize trading metrics, predict profitability, benchmark against markets, and explore trades interactively.
Jupyter Notebook (trader_sentiment_analysis.ipynb): Analyze the relationship between trade outcomes and the Fear and Greed Index using machine learning.


Features

Click to Expand: Dashboard Features


Performance Analysis:

PnL distribution histograms by profit/loss.
Cumulative PnL over time with trend visualization.
Breakdown by trading strategy, market sentiment, and Fear and Greed classification.
Time-based performance (hourly, daily, weekly, monthly).


Predictive Modeling:

Random Forest classifier to predict trade profitability.
Features include trade duration, size, timing, sentiment, strategy, and Fear and Greed score.
Interactive trade outcome predictor with confidence gauge.


Market Benchmarking:

Compare returns against assets like BTC-USD, ETH-USD, and SPY.
Metrics: Sharpe Ratio, Sortino Ratio, Max Drawdown, and correlation analysis.


Trade Explorer:

Filter trades by date, symbol, strategy, sentiment, and PnL.
Sortable table with columns for date, symbol, side, PnL, etc.
Download filtered trades as CSV.






Click to Expand: Notebook Features


Sentiment Analysis:

Merges trader data with Fear and Greed Index data.
Analyzes PnL statistics by sentiment classification (Extreme Fear, Fear, Neutral, Greed, Extreme Greed).


Machine Learning:

Random Forest model to predict trade success based on sentiment score and trade size.
Visualizes feature importance for predictive factors.






Prerequisites
Before running the project, ensure you have:

Python: Version 3.8 or higher.
Data Files:
data/trader_data.csv: Trading data with columns like Coin, Execution Price, Size Tokens, Side, Closed PnL, Fee, Timestamp IST (format: DD-MM-YYYY HH:MM).
data/fear_greed_index.csv: Fear and Greed Index data with date (format: YYYY-MM-DD), value, and classification.


Dependencies: Listed in requirements.txt.


Installation
Follow these steps to set up the project:

Clone or Download:
git clone <repository-url>
cd <project-directory>


Create a Virtual Environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Prepare Data:

Place trader_data.csv and fear_greed_index.csv in the data/ directory.
Ensure Closed PnL in trader_data.csv has non-zero values for predictive modeling.
Verify date formats align for accurate data merging.




Usage
Running the Streamlit Dashboard

Navigate to the project directory.
Launch the dashboard:streamlit run dashboard.py


Open the URL (e.g., http://localhost:8501) in a browser to interact with the dashboard.

Tips:

Use sidebar filters to narrow down trades by date, symbol, strategy, sentiment, or PnL.
Enable/disable predictive analytics in the sidebar.
Download trade data from the Trade Explorer tab.

Running the Jupyter Notebook

Start Jupyter Notebook:jupyter notebook


Open trader_sentiment_analysis.ipynb in the browser.
Run cells sequentially to load data, preprocess, and analyze sentiment.

Tips:

Check for missing values or date mismatches in the notebook output.
Extend the model by adding features like Leverage if available.


Project Structure
project/
├── data/
│   ├── trader_data.csv           # Trading data
│   ├── fear_greed_index.csv      # Fear and Greed Index data
├── dashboard.py                  # Streamlit dashboard
├── trader_sentiment_analysis.ipynb # Sentiment analysis notebook
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies


Dependencies



Package
Version
Purpose



streamlit
1.23.1
Web interface for visualization


pandas
1.5.3
Data manipulation


numpy
1.24.3
Numerical computations


plotly
5.14.1
Interactive plotting


scikit-learn
1.2.2
Machine learning


yfinance
0.2.18
Market data retrieval


matplotlib
3.7.1
Plotting (notebook)


seaborn
0.12.2
Statistical visualizations (notebook)


jupyter
1.0.0
Jupyter notebook support


Install with:
pip install -r requirements.txt


Data Notes

trader_data.csv:

Required columns: Coin, Execution Price, Size Tokens, Side, Closed PnL, Fee, Timestamp IST.
Non-zero Closed PnL values are critical for predictive modeling.
Timestamp IST should be in DD-MM-YYYY HH:MM format.


fear_greed_index.csv:

Required columns: date, value, classification.
date should be in YYYY-MM-DD format to match trader_data.csv after conversion.
Valid classification values: Extreme Fear, Fear, Neutral, Greed, Extreme Greed.


Common Issues:

Mismatched dates between files can lead to missing Fear and Greed data. Check date formats and ranges.
Zero PnL values limit model performance. Update trader_data.csv with varied PnL.
Default trade duration is 60 minutes in dashboard.py. Adjust if actual durations are available.




Contributing
We welcome contributions! To contribute:

Fork the repository.
Create a feature branch (git checkout -b feature-name).
Commit changes (git commit -m "Add feature").
Push to the branch (git push origin feature-name).
Open a pull request.

Report issues or suggest features via the repository's issue tracker.

License
This project is licensed under the MIT License.

Troubleshooting

Click to Expand: Common Issues and Solutions


Dashboard shows "No trades match your filter criteria":

Check filter settings in the sidebar (e.g., date range, symbols).
Ensure trader_data.csv contains valid data matching filter criteria.


Predictive model fails with "Not enough data variation":

Verify Closed PnL in trader_data.csv has non-zero values.
Add varied trade outcomes to enable binary classification.


Fear and Greed data not merging:

Ensure fear_greed_index.csv dates (YYYY-MM-DD) align with trader_data.csv dates (DD-MM-YYYY) after conversion.
Check for missing or invalid dates in both files.


Dependency installation errors:

Ensure Python 3.8+ is installed.
Try upgrading pip: pip install --upgrade pip.
Install dependencies individually to identify conflicts.






Contact
For questions or support, open an issue in the repository or contact the maintainers.
Back to Top
