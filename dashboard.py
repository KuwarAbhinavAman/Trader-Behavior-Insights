import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import yfinance as yf
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="Advanced Trading Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #0E1117;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .stSelectbox, .stMultiselect, .stDateInput {
        margin-bottom: 15px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 25px;
        background-color: #0E1117;
        border-radius: 10px 10px 0 0;
        gap: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #262730;
    }
</style>
""", unsafe_allow_html=True)

# Load trade data
@st.cache_data
def load_trade_data():
    try:
        df = pd.read_csv('data/trader_data.csv')
        # Convert Timestamp IST to datetime
        df['date'] = pd.to_datetime(df['Timestamp IST'], dayfirst=True, format='mixed', errors='coerce')
        # Drop rows with invalid dates
        df = df.dropna(subset=['date'])
        # Rename columns to match dashboard schema
        df = df.rename(columns={
            'Coin': 'symbol',
            'Execution Price': 'entry_price',
            'Size Tokens': 'quantity',
            'Closed PnL': 'pnl',
            'Fee': 'fee',
            'Side': 'side'
        })
        # Derive exit_price
        df['exit_price'] = df['entry_price'] + (df['pnl'] + df['fee']) / df['quantity']
        # Adjust PnL for sell trades
        df.loc[df['side'] == 'SELL', 'pnl'] = -df['pnl']
        df.loc[df['side'] == 'SELL', 'exit_price'] = df['entry_price'] - (df['pnl'] + df['fee']) / df['quantity']
        # Calculate additional columns
        df['return_pct'] = (df['exit_price'] - df['entry_price']) / df['entry_price'] * 100
        df['profit'] = df['pnl'] > 0
        df['hour_of_day'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.day_name()
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['month'] = df['date'].dt.month_name()
        # Add synthetic sentiment and strategy
        sentiments = ['Bullish', 'Bearish', 'Neutral']
        strategies = ['Scalping', 'Swing', 'Position', 'Arbitrage', 'Day']
        np.random.seed(42)
        df['sentiment'] = np.random.choice(sentiments, len(df), p=[0.45, 0.35, 0.2])
        df['strategy'] = np.random.choice(strategies, len(df), p=[0.3, 0.4, 0.15, 0.1, 0.05])
        return df
    except Exception as e:
        st.error(f"Error loading trader data: {str(e)}")
        return pd.DataFrame()

# Load Fear and Greed Index data
@st.cache_data
def load_fear_greed_data():
    try:
        fear_greed = pd.read_csv('data/fear_greed_index.csv')
        # Convert date column to datetime with flexible parsing
        fear_greed['date'] = pd.to_datetime(fear_greed['date'], format='mixed', dayfirst=True, errors='coerce')
        # Drop rows with invalid dates
        fear_greed = fear_greed.dropna(subset=['date'])
        # Convert to date for merging
        fear_greed['date'] = fear_greed['date'].dt.date
        fear_greed = fear_greed[['date', 'value', 'classification']].rename(columns={'value': 'fear_greed_score'})
        if fear_greed.empty:
            st.warning("Fear and Greed Index data is empty or contains no valid dates.")
        return fear_greed
    except Exception as e:
        st.error(f"Error loading Fear and Greed Index data: {str(e)}")
        return pd.DataFrame()

# Load market data
@st.cache_data
def load_market_data(symbols, start_date, end_date):
    try:
        data = yf.download(
            symbols,
            start=start_date,
            end=end_date,
            progress=False
        )['Adj Close']
        
        if data.empty:
            st.warning(f"No market data found for {symbols} between {start_date} and {end_date}")
            return pd.DataFrame()
            
        return data.pct_change().dropna()
    except Exception as e:
        st.error(f"Error loading market data: {str(e)}")
        return pd.DataFrame()

# Feature engineering for predictive model
def prepare_model_data(df):
    # Ensure all required columns exist
    df['trade_size'] = df['entry_price'] * df['quantity']
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day']/24)
    
    # Define all possible categories for one-hot encoding
    sentiments = ['Bullish', 'Bearish', 'Neutral']
    strategies = ['Scalping', 'Swing', 'Position', 'Arbitrage', 'Day']
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Create dummy columns for all possible categories
    sentiment_dummies = pd.get_dummies(df['sentiment'], prefix='sentiment').reindex(columns=[f'sentiment_{s}' for s in sentiments], fill_value=0)
    strategy_dummies = pd.get_dummies(df['strategy'], prefix='strategy').reindex(columns=[f'strategy_{s}' for s in strategies], fill_value=0)
    day_dummies = pd.get_dummies(df['day_of_week'], prefix='day_of_week').reindex(columns=[f'day_of_week_{d}' for d in days_of_week], fill_value=0)
    
    # Combine all features
    features = pd.concat([
        df[['duration_minutes', 'trade_size', 'hour_sin', 'hour_cos', 'fear_greed_score']],
        sentiment_dummies.drop(columns=['sentiment_Bearish']),  # Drop one category as reference
        strategy_dummies.drop(columns=['strategy_Scalping']),   # Drop one category as reference
        day_dummies.drop(columns=['day_of_week_Friday'])        # Drop one category as reference
    ], axis=1)
    
    target = df['profit']
    
    return features, target

# Train predictive model
def train_model(features, target):
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return model, accuracy, pd.DataFrame(report).transpose(), y_test, y_proba

# Main application
def main():
    st.title("🚀 Advanced Trading Analytics Dashboard")
    st.markdown("""
    <div class="title">Advanced Trading Analytics Dashboard</div>
    <div class="subtitle">Analyze performance, predict outcomes, and optimize strategies with Fear and Greed Index</div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_trade_data()
    if df.empty:
        st.error("No trade data loaded. Please check the data file and try again.")
        st.stop()
    
    fear_greed = load_fear_greed_data()
    start_date = df['date'].min().date()
    end_date = df['date'].max().date()
    
    # Merge trade data with Fear and Greed Index
    df['trade_date'] = df['date'].dt.date
    if not fear_greed.empty:
        df = pd.merge(
            df,
            fear_greed,
            left_on='trade_date',
            right_on='date',
            how='left'
        ).drop(columns=['date_y'], errors='ignore')
        df = df.rename(columns={'date_x': 'date'})
        matched_trades = df['fear_greed_score'].notna().sum()
        if matched_trades == 0:
            st.warning("No trades matched with Fear and Greed Index data. Ensure fear_greed_index.csv has dates matching trader_data.csv (e.g., 02-12-2024).")
    else:
        st.warning("Fear and Greed Index data is empty. Proceeding without it.")
        df['fear_greed_score'] = np.nan
        df['classification'] = 'Neutral'
    
    # Handle missing Fear and Greed data
    df['fear_greed_score'] = df['fear_greed_score'].fillna(df['fear_greed_score'].mean() if df['fear_greed_score'].notna().any() else 50)
    df['classification'] = df['classification'].fillna('Neutral')
    
    # Check for zero PnL
    if (df['pnl'] == 0).all():
        st.warning("All trades have zero PnL. Predictive modeling and performance metrics may be limited. Add varied PnL data to trader_data.csv.")
    
    # Add duration_minutes (since trader_data.csv lacks it)
    df['duration_minutes'] = 60  # Default value; adjust based on your needs
    
    # Sidebar filters
    with st.sidebar:
        st.header("🔍 Filters")
        
        date_range = st.date_input(
            "Date Range",
            value=(start_date, end_date),
            min_value=start_date,
            max_value=end_date
        )
        
        if len(date_range) != 2:
            st.warning("Please select both start and end dates")
            st.stop()
            
        selected_symbols = st.multiselect(
            "Cryptocurrencies",
            options=df['symbol'].unique(),
            default=df['symbol'].unique()[:3]
        )
        
        selected_strategies = st.multiselect(
            "Trading Strategies",
            options=df['strategy'].unique(),
            default=df['strategy'].unique()[:2]
        )
        
        selected_sentiments = st.multiselect(
            "Market Sentiments",
            options=df['sentiment'].unique(),
            default=df['sentiment'].unique()
        )
        
        selected_classifications = st.multiselect(
            "Fear and Greed Classifications",
            options=df['classification'].unique(),
            default=df['classification'].unique()
        )
        
        pnl_range = st.slider(
            "PnL Range (USD)",
            float(df['pnl'].min()),
            float(df['pnl'].max()),
            (float(df['pnl'].quantile(0.1)), float(df['pnl'].quantile(0.9)))
        )
        
        st.markdown("---")
        st.markdown("**Model Settings**")
        model_toggle = st.checkbox("Enable Predictive Analytics", True)
        st.markdown("---")
        st.markdown("Built with ❤️ using Streamlit")
    
    # Apply filters
    filtered_df = df[
        (df['symbol'].isin(selected_symbols)) &
        (df['date'].dt.date >= date_range[0]) &
        (df['date'].dt.date <= date_range[1]) &
        (df['strategy'].isin(selected_strategies)) &
        (df['sentiment'].isin(selected_sentiments)) &
        (df['classification'].isin(selected_classifications)) &
        (df['pnl'] >= pnl_range[0]) &
        (df['pnl'] <= pnl_range[1])
    ].copy()
    
    if filtered_df.empty:
        st.warning("No trades match your filter criteria. Please adjust filters.")
        st.stop()
    
    # Main metrics
    st.header("📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Trades", len(filtered_df))
    with col2:
        win_rate = filtered_df['profit'].mean()
        st.metric("Win Rate", f"{win_rate:.1%}",
                 delta=f"{(win_rate - 0.5):.1%} vs baseline")
    with col3:
        avg_pnl = filtered_df['pnl'].mean()
        st.metric("Avg. PnL", f"${avg_pnl:,.2f}",
                 delta_color="inverse" if avg_pnl < 0 else "normal")
    with col4:
        total_pnl = filtered_df['pnl'].sum()
        st.metric("Total PnL", f"${total_pnl:,.2f}",
                 delta_color="inverse" if total_pnl < 0 else "normal")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Performance Analysis", 
        "🔮 Predictive Modeling", 
        "📊 Benchmarking", 
        "🔍 Trade Explorer"
    ])
    
    with tab1:
        st.subheader("Performance Breakdown")
        
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.histogram(
                filtered_df, 
                x='pnl', 
                nbins=50,
                title='PnL Distribution',
                color='profit',
                color_discrete_map={True: '#00cc96', False: '#ef553b'}
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            cumulative_pnl = filtered_df.sort_values('date')['pnl'].cumsum()
            fig2 = px.line(
                x=filtered_df.sort_values('date')['date'],
                y=cumulative_pnl,
                title='Cumulative PnL Over Time'
            )
            fig2.add_hline(y=0, line_dash="dash", line_color="white")
            st.plotly_chart(fig2, use_container_width=True)
        
        st.subheader("Performance by Attributes")
        col1, col2 = st.columns(2)
        with col1:
            strategy_perf = filtered_df.groupby('strategy').agg({
                'pnl': ['mean', 'count'],
                'profit': 'mean'
            }).sort_values(('pnl', 'mean'), ascending=False)
            
            if not strategy_perf.empty:
                fig3 = px.bar(
                    strategy_perf,
                    x=strategy_perf.index,
                    y=('pnl', 'mean'),
                    title='Average PnL by Strategy',
                    color=('profit', 'mean'),
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.warning("No strategy performance data available.")
        
        with col2:
            classification_perf = filtered_df.groupby('classification').agg({
                'pnl': ['mean', 'count'],
                'profit': 'mean'
            }).reset_index()
            classification_perf.columns = ['classification', 'pnl_mean', 'pnl_count', 'profit_mean']
            classification_perf = classification_perf.dropna(subset=['pnl_mean'])
            
            if not classification_perf.empty:
                if len(classification_perf['classification']) > 1 or classification_perf['classification'].iloc[0] != 'Neutral':
                    fig4 = px.bar(
                        classification_perf,
                        x='classification',
                        y='pnl_mean',
                        title='Average PnL by Fear and Greed Classification',
                        color='profit_mean',
                        color_continuous_scale='RdYlGn'
                    )
                    st.plotly_chart(fig4, use_container_width=True)
                else:
                    st.warning("Only 'Neutral' classification available due to missing Fear and Greed data. Update fear_greed_index.csv with matching dates.")
            else:
                st.warning("No valid Fear and Greed classification performance data available. Check data or filters.")
        
        st.subheader("Time-Based Performance")
        time_group = st.radio(
            "Group by:",
            ["Hour of Day", "Day of Week", "Week of Year", "Month"],
            horizontal=True,
            key="time_group"
        )
        
        if time_group == "Hour of Day":
            group_col = 'hour_of_day'
        elif time_group == "Day of Week":
            group_col = 'day_of_week'
            filtered_df[group_col] = pd.Categorical(
                filtered_df[group_col],
                categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                ordered=True
            )
        elif time_group == "Week of Year":
            group_col = 'week_of_year'
        else:
            group_col = 'month'
            filtered_df[group_col] = pd.Categorical(
                filtered_df[group_col],
                categories=['January', 'February', 'March', 'April', 'May', 'June', 
                          'July', 'August', 'September', 'October', 'November', 'December'],
                ordered=True
            )
        
        # Create a properly aggregated DataFrame for plotting
        time_perf = filtered_df.groupby(group_col, observed=True).agg({
            'pnl': ['mean', 'count'],
            'profit': 'mean'
        }).reset_index()
        
        # Flatten multi-index columns
        time_perf.columns = ['_'.join(col).strip('_') for col in time_perf.columns.values]
        
        # Plot average PnL
        fig5 = px.line(
            time_perf,
            x=group_col,
            y='pnl_mean',
            title=f'Average PnL by {time_group}',
            markers=True
        )
        
        # Plot win rate
        fig6 = px.bar(
            time_perf,
            x=group_col,
            y='profit_mean',
            title=f'Win Rate by {time_group}',
            color='profit_mean',
            color_continuous_scale='RdYlGn',
            range_color=[0, 1]
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig5, use_container_width=True)
        with col2:
            st.plotly_chart(fig6, use_container_width=True)
    
    with tab2:
        if not model_toggle:
            st.warning("Predictive analytics is disabled in the sidebar settings")
            st.stop()
            
        st.subheader("Predictive Trade Analytics")
        
        with st.expander("About the Model", expanded=False):
            st.markdown("""
            This predictive model uses a Random Forest classifier to predict whether a trade will be profitable based on:
            - Trade characteristics (duration, size)
            - Timing (hour of day, day of week)
            - Market sentiment
            - Trading strategy
            - Fear and Greed Index score
            """)
        
        with st.spinner("Training predictive model..."):
            features, target = prepare_model_data(filtered_df)
            
            if len(target.unique()) < 2:
                st.error("Not enough data variation to train model. Ensure trader_data.csv has non-zero PnL values.")
                st.stop()
                
            model, accuracy, report, y_test, y_proba = train_model(features, target)
            
        st.subheader("Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{accuracy:.1%}")
        with col2:
            baseline = target.mean()
            st.metric("Improvement Over Baseline", 
                     f"{(accuracy - max(baseline, 1-baseline)):.1%}")
        
        st.subheader("Feature Importance")
        feat_importance = pd.DataFrame({
            'Feature': features.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig7 = px.bar(
            feat_importance.head(10),
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top 10 Most Important Features'
        )
        st.plotly_chart(fig7, use_container_width=True)
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        fig8 = go.Figure()
        fig8.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.2f})',
            line=dict(color='#00cc96', width=2)
        ))
        fig8.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Chance',
            line=dict(color='white', dash='dash')
        ))
        fig8.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            showlegend=True
        )
        st.plotly_chart(fig8, use_container_width=True)
        
        st.subheader("Trade Outcome Predictor")
        col1, col2 = st.columns(2)
        with col1:
            duration = st.slider("Duration (minutes)", 1, 1440, 60)
            trade_size = st.slider("Trade Size (USD)", 10, 10000, 1000)
            hour = st.slider("Hour of Day", 0, 23, 12)
        with col2:
            sentiment = st.selectbox("Market Sentiment", ['Bullish', 'Bearish', 'Neutral'])
            strategy = st.selectbox("Trading Strategy", ['Scalping', 'Swing', 'Position', 'Arbitrage', 'Day'])
            day_of_week = st.selectbox("Day of Week", 
                ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            fear_greed_score = st.slider("Fear and Greed Score", 0, 100, 50)
        
        # Create input data with all possible dummy columns
        input_data = {
            'duration_minutes': duration,
            'trade_size': trade_size,
            'hour_sin': np.sin(2 * np.pi * hour/24),
            'hour_cos': np.cos(2 * np.pi * hour/24),
            'fear_greed_score': fear_greed_score
        }
        
        # Add sentiment dummies
        for s in ['Bullish', 'Neutral']:
            input_data[f'sentiment_{s}'] = 1 if sentiment == s else 0
            
        # Add strategy dummies
        for s in ['Swing', 'Position', 'Arbitrage', 'Day']:
            input_data[f'strategy_{s}'] = 1 if strategy == s else 0
            
        # Add day of week dummies
        for d in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Saturday', 'Sunday']:
            input_data[f'day_of_week_{d}'] = 1 if day_of_week == d else 0
        
        input_df = pd.DataFrame([input_data])[features.columns]
        
        if st.button("Predict Outcome"):
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][1]
            
            st.metric(
                "Predicted Outcome", 
                "✅ Profitable" if prediction else "❌ Unprofitable",
                delta=f"{proba:.1%} confidence"
            )
            
            fig9 = go.Figure()
            fig9.add_trace(go.Indicator(
                mode="gauge+number",
                value=proba*100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Probability of Profit"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            st.plotly_chart(fig9, use_container_width=True)
    
    with tab3:
        st.subheader("Market Benchmarking")
        
        benchmark_symbols = st.multiselect(
            "Select Assets to Compare Against",
            options=['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'DOGE-USD', 'MATIC-USD', 'SPY', 'QQQ'],
            default=['BTC-USD', 'ETH-USD', 'SPY']
        )
        
        if benchmark_symbols:
            with st.spinner("Loading market data..."):
                market_data = load_market_data(
                    benchmark_symbols,
                    start_date=date_range[0],
                    end_date=date_range[1]
                )
                
            if market_data.empty:
                st.warning("No market data available for comparison")
            else:
                trader_daily = filtered_df.groupby(filtered_df['date'].dt.date)['return_pct'].mean()
                comparison = pd.DataFrame({
                    'Your Strategy': trader_daily,
                    **{sym: market_data[sym]*100 for sym in benchmark_symbols}
                }).dropna()
                
                st.subheader("Cumulative Returns Comparison")
                fig10 = px.line(
                    (1 + comparison/100).cumprod(),
                    title="Growth of $1 Investment",
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                fig10.update_yaxes(tickprefix="$", type="log")
                st.plotly_chart(fig10, use_container_width=True)
                
                st.subheader("Risk-Adjusted Performance")
                sharpe_ratio = comparison.mean() / comparison.std() * np.sqrt(365)
                sortino_ratio = comparison.mean() / comparison[comparison < 0].std() * np.sqrt(365)
                max_drawdown = (1 + comparison/100).cumprod().div(
                    (1 + comparison/100).cumprod().cummax()
                ).sub(1).min()
                
                metrics_df = pd.DataFrame({
                    'Asset': sharpe_ratio.index,
                    'Sharpe Ratio': sharpe_ratio.values,
                    'Sortino Ratio': sortino_ratio.values,
                    'Max Drawdown': max_drawdown.values
                }).melt(id_vars='Asset', var_name='Metric')
                
                fig11 = px.bar(
                    metrics_df,
                    x='Asset',
                    y='value',
                    color='Metric',
                    barmode='group',
                    title='Performance Metrics Comparison'
                )
                st.plotly_chart(fig11, use_container_width=True)
                
                st.subheader("Correlation Analysis")
                corr_matrix = comparison.corr()
                
                fig12 = px.imshow(
                    corr_matrix,
                    text_auto=".2f",
                    color_continuous_scale='RdYlGn',
                    range_color=[-1, 1],
                    title="Correlation Between Your Strategy and Markets"
                )
                st.plotly_chart(fig12, use_container_width=True)
    
    with tab4:
        st.subheader("Trade Explorer")
        
        col1, col2 = st.columns(2)
        with col1:
            show_profitable = st.checkbox("Show only profitable trades", False)
        with col2:
            sort_by = st.selectbox("Sort by", 
                                 ['Date (Newest)', 'Date (Oldest)', 'PnL (High)', 'PnL (Low)', 'Duration'])
        
        explorer_df = filtered_df.copy()
        if show_profitable:
            explorer_df = explorer_df[explorer_df['profit']]
            
        if sort_by == 'Date (Newest)':
            explorer_df = explorer_df.sort_values('date', ascending=False)
        elif sort_by == 'Date (Oldest)':
            explorer_df = explorer_df.sort_values('date', ascending=True)
        elif sort_by == 'PnL (High)':
            explorer_df = explorer_df.sort_values('pnl', ascending=False)
        elif sort_by == 'PnL (Low)':
            explorer_df = explorer_df.sort_values('pnl', ascending=True)
        elif sort_by == 'Duration':
            explorer_df = explorer_df.sort_values('duration_minutes', ascending=False)
        
        st.dataframe(
            explorer_df[[
                'date', 'symbol', 'side', 'strategy', 'sentiment', 'classification', 'fear_greed_score',
                'entry_price', 'exit_price', 'quantity', 'pnl', 'return_pct', 'duration_minutes', 'profit'
            ]].rename(columns={
                'date': 'Date',
                'symbol': 'Symbol',
                'side': 'Side',
                'strategy': 'Strategy',
                'sentiment': 'Sentiment',
                'classification': 'Fear/Greed Classification',
                'fear_greed_score': 'Fear/Greed Score',
                'entry_price': 'Entry Price',
                'exit_price': 'Exit Price',
                'quantity': 'Quantity',
                'pnl': 'PnL',
                'return_pct': 'Return %',
                'duration_minutes': 'Duration (min)',
                'profit': 'Profitable'
            }),
            column_config={
                'Date': st.column_config.DatetimeColumn("Date"),
                'Entry Price': st.column_config.NumberColumn("Entry", format="$%.2f"),
                'Exit Price': st.column_config.NumberColumn("Exit", format="$%.2f"),
                'PnL': st.column_config.NumberColumn("PnL", format="$%.2f"),
                'Return %': st.column_config.NumberColumn("Return %", format="%.2f%%"),
                'Fear/Greed Score': st.column_config.NumberColumn("Fear/Greed Score", format="%.0f"),
                'Profitable': st.column_config.CheckboxColumn("Profitable")
            },
            hide_index=True,
            use_container_width=True,
            height=600
        )
        
        csv = explorer_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Trade Data as CSV",
            data=csv,
            file_name=f"trades_{date_range[0]}_{date_range[1]}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()