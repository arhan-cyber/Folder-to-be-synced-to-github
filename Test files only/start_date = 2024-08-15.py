start_date = 2024-08-15
end_date = 2024-08-31
cols = fixed_dates[start_date]
optimization_start_date = 2023-08-15
optimization_end_date = 2023=08-14
optimization_df = new_df[optimization_start_date:optimization_end_date]['Adj Close'][cols]
success = False
try:
    weights = optimize_weights(prices=optimization_df,
                                   lower_bound=round(1/(len(optimization_df.columns)*2),3))

    weights = pd.DataFrame(weights, index=pd.Series(0))
            
    success = True
except:
    print(f'Max Sharpe Optimization failed for {start_date}, Continuing with Equal-Weights')
if success==False:
    weights = pd.DataFrame([1/len(optimization_df.columns) for i in range(len(optimization_df.columns))],
                                     index=optimization_df.columns.tolist(),
                                     columns=pd.Series(0)).T
        
temp_df = returns_dataframe[start_date:end_date]

temp_df = temp_df.stack().to_frame('return').reset_index(level=0)\
                   .merge(weights.stack().to_frame('weight').reset_index(level=0, drop=True),
                          left_index=True,
                          right_index=True)\
                   .reset_index().set_index(['Date', 'index']).unstack().stack()

temp_df.index.names = ['date', 'ticker']

temp_df['weighted_return'] = temp_df['return']*temp_df['weight']

temp_df = temp_df.groupby(level=0)['weighted_return'].sum().to_frame('Strategy Return')

portfolio_df = pd.concat([portfolio_df, temp_df], axis=0)


print (portfolio_df)