from statsmodels.regression.rolling import RollingOLS
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import pandas_ta
import warnings
warnings.filterwarnings('ignore')

sussy = pd.read_excel(r'C:\Users\ajay_\Downloads\Folder to be synced to github\Finance\your_file.xlsx', sheet_name='Sheet1', usecols='A', skiprows=2, nrows=50)
print (sussy)
tickers = sussy.iloc[:, 0].tolist()  
tickers = [ticker + ".NS" for ticker in tickers]

end_date = '2024-09-27'
start_date = pd.to_datetime(end_date)-pd.DateOffset(365*8)


df_new = yf.download(tickers=tickers,
                 start=start_date,
                 end=end_date).stack()

df_new.index.names = ['date', 'ticker']

df_new.columns = df_new.columns.str.lower()


df=df_new

df['garman_klass_vol'] = ((np.log(df['high'])-np.log(df['low']))**2)/2-(2*np.log(2)-1)*((np.log(df['adj close'])-np.log(df['open']))**2)

df['rsi'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))

df['bb_low'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,0])
                                                          
df['bb_mid'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,1])
                                                          
df['bb_high'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,2])

def compute_atr(stock_data):
    atr = pandas_ta.atr(high=stock_data['high'],
                        low=stock_data['low'],
                        close=stock_data['close'],
                        length=14)
    return atr.sub(atr.mean()).div(atr.std())

df['atr'] = df.groupby(level=1, group_keys=False).apply(compute_atr)

def compute_macd(close):
    macd = pandas_ta.macd(close=close, length=20).iloc[:,0]
    return macd.sub(macd.mean()).div(macd.std())

df['macd'] = df.groupby(level=1, group_keys=False)['adj close'].apply(compute_macd)

df['dollar_volume'] = (df['adj close']*df['volume'])/1e6

print (df)
last_cols = [c for c in df.columns.unique(0) if c not in ['dollar_volume', 'volume', 'open',
                                                          'high', 'low', 'close']]

data = (pd.concat([df.unstack('ticker')['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume'),
                   df.unstack()[last_cols].resample('M').last().stack('ticker')],
                  axis=1)).dropna()

data['dollar_volume'] = (data.loc[:, 'dollar_volume'].unstack('ticker').rolling(5*12, min_periods=12).mean().stack())

data['dollar_vol_rank'] = (data.groupby('date')['dollar_volume'].rank(ascending=False))

data = data[data['dollar_vol_rank']<40].drop(['dollar_volume', 'dollar_vol_rank'], axis=1)


def calculate_returns(df):

    outlier_cutoff = 0.005

    lags = [1, 2, 3, 6, 9, 12]

    for lag in lags:

        df[f'return_{lag}m'] = (df['adj close']
                              .pct_change(lag)
                              .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                     upper=x.quantile(1-outlier_cutoff)))
                              .add(1)
                              .pow(1/lag)
                              .sub(1))
    return df
    
    
data = data.groupby(level=1, group_keys=False).apply(calculate_returns).dropna()



print (data)

# Assuming 'date' is the first level of your MultiIndex:
if isinstance(data.index, pd.MultiIndex):
    # Extract the 'date' level (assuming it's the first level)
    date_index = data.index.get_level_values('date')  # or use 0 if it's the first level
    print(date_index.tz)  # This will show None if tz-naive, or the timezone if tz-aware
else:
    print(data.index.tz)  # If it's not MultiIndex, check tz directly



factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3',
                               'famafrench',
                               start='2010')[0].drop('RF', axis=1)

factor_data.index = factor_data.index.to_timestamp()

factor_data = factor_data.resample('M').last().div(100)

factor_data.index.name = 'date'

# # Assuming 'date' is the first level of your MultiIndex:
# if isinstance(factor_data.index, pd.MultiIndex):
#     # Extract the 'date' level (assuming it's the first level)
#     date_index = factor_data.index.get_level_values('date')  # or use 0 if it's the first level
#     print(date_index.tz)  # This will show None if tz-naive, or the timezone if tz-aware
# else:
#     print(factor_data.index.tz)  # If it's not MultiIndex, check tz directly




# Convert the date index to UTC if it's tz-naive
if isinstance(factor_data.index, pd.MultiIndex):
    # Extract the 'date' level
    date_index = factor_data.index.get_level_values('date')
else:
    date_index = factor_data.index

# Check if the index is timezone-naive
if date_index.tz is None:
    # Localize to UTC
    date_index = date_index.tz_localize('UTC')
    # If you want to replace the index in factor_data
    if isinstance(factor_data.index, pd.MultiIndex):
        factor_data.index = factor_data.index.set_levels([date_index] + list(factor_data.index.levels[1:]))
    else:
        factor_data.index = date_index

# Check the updated timezone
print(factor_data.index.tz)


factor_data = factor_data.join(data['return_1m']).sort_index()
observations = factor_data.groupby(level=1).size()

valid_stocks = observations[observations >= 10]

factor_data = factor_data[factor_data.index.get_level_values('ticker').isin(valid_stocks.index)]

print (factor_data)

betas = (factor_data.groupby(level=1,
                            group_keys=False)
         .apply(lambda x: RollingOLS(endog=x['return_1m'], 
                                     exog=sm.add_constant(x.drop('return_1m', axis=1)),
                                     window=min(24, x.shape[0]),
                                     min_nobs=len(x.columns)+1)
         .fit(params_only=True)
         .params
         .drop('const', axis=1)))

print (betas) 


factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

data = (data.join(betas.groupby('ticker').shift()))

data.loc[:, factors] = data.groupby('ticker', group_keys=False)[factors].apply(lambda x: x.fillna(x.mean()))

data = data.drop('adj close', axis=1)

data = data.dropna()

data.info()

from sklearn.cluster import KMeans

# You don't need to define `initial_centroids` manually. The K-Means++ is selected with init='k-means++'

# def get_clusters(df):
#     kmeans = KMeans(
#         n_clusters=4,  # Number of clusters
#         random_state=0,  # Ensures reproducibility
#         init='k-means++'  # K-Means++ initialization
#     )
    
#     # Fit the model and assign cluster labels
#     df['cluster'] = kmeans.fit(df).labels_
    
#     return df

target_rsi_values = [-0.9, -0.3, 0.3, 0.9]

initial_centroids = np.zeros((len(target_rsi_values), 18))

initial_centroids[:, 6] = target_rsi_values

print (initial_centroids)
def get_clusters(df):
    df['cluster'] = KMeans(n_clusters=4,
                           random_state=0,
                           init=initial_centroids).fit(df).labels_
    return df
# Assuming `data` is already defined
data = data.dropna().groupby('date', group_keys=False).apply(get_clusters)
print(data)



# from sklearn.cluster import KMeans
# from sklearn.cluster import AgglomerativeClustering

# # Perform hierarchical clustering
# hc = AgglomerativeClustering(n_clusters=4)
# hc.fit(data)

# # Get cluster centers as initial centroids
# initial_centroids = data.groupby(hc.labels_).mean().values

# data = data.drop('cluster', axis=1, errors='ignore')


# def get_clusters(df):
#     df['cluster'] = KMeans(n_clusters=4,
#                            random_state=0,
#                            init=initial_centroids).fit(df).labels_
#     return df

# data = data.dropna().groupby('date', group_keys=False).apply(get_clusters)
# print (data)


def plot_clusters(data):

    cluster_0 = data[data['cluster']==0]
    cluster_1 = data[data['cluster']==1]
    cluster_2 = data[data['cluster']==2]
    cluster_3 = data[data['cluster']==3]

    plt.scatter(cluster_0.iloc[:,0] , cluster_0.iloc[:,6] , color = 'red', label='cluster 0')
    plt.scatter(cluster_1.iloc[:,0] , cluster_1.iloc[:,6] , color = 'green', label='cluster 1')
    plt.scatter(cluster_2.iloc[:,0] , cluster_2.iloc[:,6] , color = 'blue', label='cluster 2')
    plt.scatter(cluster_3.iloc[:,0] , cluster_3.iloc[:,6] , color = 'black', label='cluster 3')
    
    plt.legend()
    plt.show()
    return

plt.style.use('ggplot')

for i in data.index.get_level_values('date').unique().tolist():
    
    g = data.xs(i, level=0)
    
    plt.title(f'Date {i}')
    
    # plot_clusters(g)



filtered_df = data[data['cluster']==3].copy()

filtered_df = filtered_df.reset_index(level=1)

filtered_df.index = filtered_df.index+pd.DateOffset(1)

filtered_df = filtered_df.reset_index().set_index(['date', 'ticker'])

dates = filtered_df.index.get_level_values('date').unique().tolist()

fixed_dates = {}

for d in dates:
    
    fixed_dates[d.strftime('%Y-%m-%d')] = filtered_df.xs(d, level=0).index.tolist()
    
print (fixed_dates)

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

def optimize_weights(prices, lower_bound=0):
    
    returns = expected_returns.mean_historical_return(prices=prices,
                                                      frequency=252)
    
    cov = risk_models.sample_cov(prices=prices,
                                 frequency=252)
    
    ef = EfficientFrontier(expected_returns=returns,
                           cov_matrix=cov,
                           weight_bounds=(lower_bound, .1),
                           solver='SCS')
    
    weights = ef.max_sharpe()
    
    return ef.clean_weights()

stocks = data.index.get_level_values('ticker').unique().tolist()

new_df = yf.download(tickers=stocks,
                     start=data.index.get_level_values('date').unique()[0]-pd.DateOffset(months=12),
                     end=data.index.get_level_values('date').unique()[-1])

print (new_df)

import pandas as pd
import numpy as np

# Select only the "Adj Close" prices from the MultiIndex columns
adj_close_df = new_df['Adj Close']

# Calculate log returns based on "Adj Close"
returns_dataframe = np.log(adj_close_df).diff()

# Initialize an empty DataFrame to hold portfolio results
portfolio_df = pd.DataFrame()

for start_date in fixed_dates.keys():
    try:
        # Define end date as the month-end of start_date
        end_date = (pd.to_datetime(start_date) + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')

        # Get the tickers/columns for this period from fixed_dates
        cols = fixed_dates[start_date]

        # Define the optimization period (12 months prior to start_date)
        optimization_start_date = (pd.to_datetime(start_date) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')
        optimization_end_date = (pd.to_datetime(start_date) - pd.DateOffset(days=1)).strftime('%Y-%m-%d')

        # Subset the adj_close_df data for optimization (only selected columns and date range)
        optimization_df = adj_close_df.loc[optimization_start_date:optimization_end_date, cols]

        success = False
        try:
            # Optimize the portfolio weights using your function
            weights = optimize_weights(prices=optimization_df, lower_bound=round(1/(len(optimization_df.columns) * 2), 3))

            # Check if weights are in the expected shape (1 row and len(cols) columns)
            if isinstance(weights, np.ndarray):
                # If weights is an array, convert it to DataFrame
                weights = pd.DataFrame(weights, columns=cols)  # Ensure column names match tickers
            elif isinstance(weights, pd.DataFrame) and weights.shape[1] > 2:
                # If weights is a DataFrame with more than 2 columns, slice or select the relevant ones
                weights = weights.iloc[:, :2]  # Keep only the first two columns

            success = True
        except Exception as opt_e:
            print(f"Max Sharpe Optimization failed for {start_date}: {opt_e}. Continuing with Equal-Weights")

        # If optimization fails, use equal weights
        if not success:
            equal_weight = 1 / len(optimization_df.columns)
            weights = pd.DataFrame([[equal_weight] * len(optimization_df.columns)],
                                   columns=optimization_df.columns.tolist())

        # Subset returns data for the corresponding date range (only selected tickers)
        temp_df = returns_dataframe.loc[start_date:end_date, cols]

        # Check if temp_df has data for merging
        if temp_df.empty:
            print(f"No return data available for {start_date} to {end_date}")
            continue

        # Stack and reset index to prepare for merging
        temp_df = temp_df.stack().reset_index()
        # Rename the columns appropriately
        temp_df.columns = ['Date', 'ticker', 'return']

        # Check the shape of weights and adjust accordingly
        weights = weights.stack().reset_index()
        print(f"Shape of weights before renaming for {start_date}: {weights.shape}")

        if weights.shape[1] == 2:
            weights.columns = ['ticker', 'weight']  # Ensure columns are properly named
        else:
            print(f"Unexpected number of columns in weights for {start_date}: {weights.shape[1]}")
            continue  # Skip this iteration if the structure is not correct

        # Merge the returns with weights on 'ticker'
        merged_df = pd.merge(temp_df, weights, on='ticker')

        # Calculate weighted returns
        merged_df['weighted_return'] = merged_df['return'] * merged_df['weight']

        # Aggregate by date to get strategy return
        merged_df = merged_df.groupby('Date')['weighted_return'].sum().to_frame('Strategy Return')

        # Append to portfolio DataFrame
        portfolio_df = pd.concat([portfolio_df, merged_df], axis=0)

    except KeyError as ke:
        print(f"KeyError: {ke} for {start_date}")
    except ValueError as ve:
        print(f"ValueError: {ve} for {start_date}")
    except Exception as e:
        print(f"Unhandled Exception: {e} for {start_date}")

# Remove duplicate rows, if any
portfolio_df = portfolio_df.drop_duplicates()

# Display the final portfolio DataFrame
print(portfolio_df)

# from pypfopt.efficient_frontier import EfficientFrontier
# from pypfopt import risk_models
# from pypfopt import expected_returns

# def optimize_weights(prices, lower_bound=0):
    
#     returns = expected_returns.mean_historical_return(prices=prices,
#                                                       frequency=252)
    
#     cov = risk_models.sample_cov(prices=prices,
#                                  frequency=252)
    
#     ef = EfficientFrontier(expected_returns=returns,
#                            cov_matrix=cov,
#                            weight_bounds=(lower_bound, .1),
#                            solver='SCS')
    
#     weights = ef.max_sharpe()
    
#     return ef.clean_weights()


# stocks = data.index.get_level_values('ticker').unique().tolist()

# new_df = yf.download(tickers=stocks,
#                      start=data.index.get_level_values('date').unique()[0]-pd.DateOffset(months=12),
#                      end=data.index.get_level_values('date').unique()[-1])

# print (new_df)
# print (new_df.columns)



# spy = yf.download(tickers='SPY',
#                   start='2015-01-01',
#                   end=dt.date.today())

# spy_ret = np.log(spy[['Adj Close']]).diff().dropna().rename({'Adj Close':'SPY Buy&Hold'}, axis=1)

# portfolio_df = portfolio_df.merge(spy_ret,
#                                   left_index=True,
#                                   right_index=True)

# print (portfolio_df)

# import matplotlib.ticker as mtick

# plt.style.use('ggplot')

# portfolio_cumulative_return = np.exp(np.log1p(portfolio_df).cumsum())-1

# portfolio_cumulative_return[:'2024-09-27'].plot(figsize=(16,6))

# plt.title('Unsupervised Learning Trading Strategy Returns Over Time')

# plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))

# plt.ylabel('Return')

# plt.show()


