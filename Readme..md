Advanced Trading Analytics Dashboard
Project Overview
This project provides a comprehensive trading analytics dashboard built with Streamlit (dashboard.py) and a Jupyter notebook for trader sentiment analysis (trader_sentiment_analysis.ipynb). The dashboard enables users to analyze trading performance, predict trade outcomes, benchmark against market indices, and explore trade data interactively. The notebook focuses on analyzing the relationship between trading performance and market sentiment using the Fear and Greed Index.
Features

Performance Analysis: Visualize PnL distribution, cumulative PnL, and performance by trading strategy, market sentiment, and Fear and Greed Index classification.
Predictive Modeling: Use a Random Forest classifier to predict trade profitability based on trade characteristics, timing, sentiment, and market conditions.
Market Benchmarking: Compare trading performance against major cryptocurrencies and market indices (e.g., BTC-USD, SPY).
Trade Explorer: Filter and sort trades with a downloadable CSV option.
Sentiment Analysis: Analyze the impact of the Fear and Greed Index on trade outcomes using the Jupyter notebook.

Prerequisites

Python 3.8 or higher
A working installation of the dependencies listed in requirements.txt
Data files:
data/trader_data.csv: Contains trading data with columns like Coin, Execution Price, Size Tokens, Side, Closed PnL, Fee, and Timestamp IST.
data/fear_greed_index.csv: Contains Fear and Greed Index data with columns date, value, and classification.



Installation

Clone the repository or download the project files.
Create a virtual environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install the required dependencies:
pip install -r requirements.txt


Ensure the data/ directory contains the required CSV files (trader_data.csv and fear_greed_index.csv).

Usage
Running the Dashboard

Navigate to the project directory.
Run the Streamlit dashboard:
streamlit run dashboard.py


Open the provided URL (typically http://localhost:8501) in a web browser to interact with the dashboard.

Running the Jupyter Notebook

Ensure Jupyter is installed (jupyter is included in requirements.txt).
Start Jupyter Notebook:
jupyter notebook


Open trader_sentiment_analysis.ipynb in the Jupyter interface and run the cells to perform sentiment analysis.

Data Requirements

trader_data.csv: Should include columns such as Coin, Execution Price, Size Tokens, Side, Closed PnL, Fee, and Timestamp IST (format: DD-MM-YYYY HH:MM). Ensure Closed PnL contains non-zero values for meaningful predictive modeling.
fear_greed_index.csv: Must include date (format: YYYY-MM-DD), value (Fear and Greed score), and classification (e.g., Extreme Fear, Fear, Neutral, Greed, Extreme Greed). Dates should align with trader_data.csv for accurate merging.

Project Structure
project/
├── data/
│   ├── trader_data.csv
│   ├── fear_greed_index.csv
├── dashboard.py
├── trader_sentiment_analysis.ipynb
├── README.md
├── requirements.txt

Dependencies
See requirements.txt for a complete list of Python packages required.
Notes

The dashboard assumes a default trade duration of 60 minutes since trader_data.csv lacks duration data. Adjust the duration_minutes calculation in dashboard.py if actual durations are available.
If trader_data.csv contains only zero PnL values, predictive modeling and performance metrics may be limited. Update the data to include varied PnL values.
Ensure date formats in both CSV files are consistent to avoid merging issues.
The Jupyter notebook includes a basic Random Forest model. Extend it by adding more features (e.g., Leverage) if available in your dataset.

Contributing
Feel free to fork the repository, make improvements, and submit pull requests. For issues or feature requests, please open an issue on the repository.
License
This project is licensed under the MIT License.