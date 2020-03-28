#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import pdb
N = 0


# In[2]:


def sharpe(ts,rfr):
    '''
        Computes the Sharpe Ratio
    '''
    rfr = rfr/12
    try:
        sh = (ts.mean() - rfr)/ts.std()
    except ZeroDivisionError:
        pdb.set_trace()
    return sh


# In[3]:


def downside(x):
    '''
        Used in sortino() function. 
    '''
    if x < 0: return x
    else: return np.NaN


# In[4]:


def sortino(ts, rfr):
    '''
    computes the sortino ratio
    '''
    rfr = rfr/12
    try:
        s = (ts.mean()-rfr)/(ts.apply(downside).std())
    except ZeroDivisionError:
        pdb.set_trace()
    return s


# In[5]:


def annualizedSharpe(ann_mean, ann_vol, rfr):
    '''
        Computes the Annualized Sharpe Ratio
    '''
    sh = (ann_mean - rfr)/ann_vol
    return sh


# In[6]:


def maxDD(pnl_ts):
    '''computes drawdown and stores spot to plot drawdown'''
    pnl_ts = pnl_ts[pnl_ts.first_valid_index():]
    drawdowns = 1 - (pnl_ts/np.maximum.accumulate(pnl_ts))
    drawdown_list_for_plotting = pd.Series([1 if x >= max(drawdowns) else 0 for x in drawdowns], index = pnl_ts.index)
    return (max(drawdowns), drawdown_list_for_plotting)


# In[7]:


def portfolioBeta(portfolio_returns, market_returns):
    '''
    compute the portfolio beta
    '''
    temp_df = pd.DataFrame()
    temp_df['portfolio'] = portfolio_returns
    temp_df['market'] = market_returns
    portfolio_covariance = temp_df.cov()['portfolio']['market']
    beta = portfolio_covariance/market_returns.var()
    return beta


# In[8]:


def get_months(d1, d2):
    '''
    function to get the months between two dates
    '''
    return (d1.year - d2.year) * 12 + d1.month - d2.month


# In[9]:


def getStatsAll(raw_data, risk_free_rate = 0.02, plot = True, beta_plot = True, training_split_value_to_plot = 0):
    
    '''
    This takes in an entire raw set of data in a dictionary format with specific keys. 
    
    It handles 20 different strategies with different beta levels and different training splits. 
    
    This function computes the necessary performance metrics, plots the performance against the market
    and then displays/returns the metrics. 
    '''
    
    total_results_dictionary = {}
    results_dictionary = {}
    all_combined_results = {}
    all_data = {}
    beta_cap_list = []
    train_split_list = []
    beta_plot_dict = {}
    
    
    for tup in list(raw_data.keys()):
        beta_cap_list.append(tup[1])
        train_split_list.append(tup[0])
    beta_cap_list = list(set(beta_cap_list))
    train_split_list = list(set(train_split_list))

    for beta_cap in beta_cap_list:
        all_data[beta_cap] = {}
        for train_split in train_split_list:
            all_data[beta_cap][train_split] = raw_data[(train_split, beta_cap)][:-1]

    all_data_keys = list(all_data.keys())
    all_data_keys.sort()
    
    for key_beta in all_data_keys:
        raw_dict = all_data[key_beta]
        beta_c = key_beta
    
        dict_df = {}
        pnl = pd.DataFrame()
        strat_rets = pd.DataFrame()
        market_rets = pd.DataFrame()
        
        for key in list(raw_dict.keys()):
            raw_dict[round(key,3)] = raw_dict.pop(key)

        for key in raw_dict.keys():
            col_list = raw_dict[key][-1].columns
            dict_df[key] = raw_dict[key][-1]
            pnl[key] = raw_dict[key][-1][col_list[2]]
            strat_rets[key] = raw_dict[key][-1][col_list[1]]
            market_rets[key] = raw_dict[key][-1][col_list[0]]

        number_of_strategies = strat_rets.columns.tolist()
        number_of_strategies.sort()
        stats_combined = {}
        combined_results = pd.DataFrame()
        
        
        for strat in number_of_strategies:
            
            stats_dict = {} 
            cumulative_portfolio_performance = (pnl[strat]-1000000)/1000000
            cumulative_market_performance = (1 + market_rets[strat]).cumprod()-1

#             sharpe_value = sharpe(strat_rets[strat], risk_free_rate)
            
            max_drawdown, maxdrawdown_for_plotting = maxDD(pnl[strat])

            portfolio_beta = portfolioBeta(strat_rets[strat], market_rets[strat])
            treynor = (strat_rets[strat].mean() - risk_free_rate)/portfolio_beta
            
            number_of_months = get_months(strat_rets[strat].last_valid_index(), strat_rets[strat].first_valid_index())

            annualized_strategy_return = (1 + strat_rets[strat]).prod()**(12/number_of_months) - 1
            annualized_market_return = (1 + market_rets[strat]).prod()**(12/number_of_months) - 1
            
            sharpe_value = sharpe(strat_rets[strat], risk_free_rate)
            sortino_value = sortino(strat_rets[strat], risk_free_rate)

            
            monthly_strategy_volatility = strat_rets[strat].std()
            monthly_market_volatility = market_rets[strat].std()

            number_of_months = get_months(strat_rets[strat].last_valid_index(), strat_rets[strat].first_valid_index())

            annualized_strategy_volatility = monthly_strategy_volatility * np.sqrt(12)
            annualized_market_volatility = monthly_market_volatility * np.sqrt(12)

            annualized_sharpe_value = annualizedSharpe(annualized_strategy_return, annualized_strategy_volatility, risk_free_rate)

            alpha = strat_rets[strat].mean() - (risk_free_rate/12) - portfolio_beta * (market_rets[strat].mean() - (risk_free_rate/12))

            

            maxdrawdown_for_plotting  = (maxdrawdown_for_plotting* cumulative_portfolio_performance).replace(0.0, np.NaN)
            index_to_start_arrow_at = maxdrawdown_for_plotting.idxmax(skipna = True)
            next_index = maxdrawdown_for_plotting.shift(1).idxmax(skipna = True)
            height_to_start_arrow_at = maxdrawdown_for_plotting.max()

            stats_dict = {'Annualized_Strategy_Return':annualized_strategy_return,
                          'Annualized_Market_Return':annualized_market_return, 
                          'Annualized_Strategy_Volatility':annualized_strategy_volatility,
                          'Annualized_Market_Volatility':annualized_market_volatility,
                          'Annualized_Sharpe':annualized_sharpe_value,
                          'Monthly_Sharpe':sharpe_value, 'Monthly_Sortino':sortino_value, 
                          'Max_Drawdown':max_drawdown, 'Portfolio_Beta':portfolio_beta, 
                          'Alpha':alpha, 'Treynor':treynor, 
                          'CumulativePortfolioPerformance':cumulative_portfolio_performance, 
                          'MaxDrawDownForPlotting':maxdrawdown_for_plotting, 
                          'index_to_start_arrow_at':index_to_start_arrow_at, 'next_index':next_index, 
                          'height_to_start_arrow_at':height_to_start_arrow_at, 
                          'CumulativeMarketPerformance':cumulative_market_performance}

            stats_combined[strat] = stats_dict    
            
            results = pd.DataFrame.from_dict(stats_dict, orient = "index", columns = ['Training Split: ' + str(strat)])
            combined_results = pd.concat([combined_results, results], axis = 1)
            
            if strat == training_split_value_to_plot:
                beta_plot_dict[key_beta] = stats_combined[strat]
        
        colorlist = ['C1', 'C2', 'C3', 'C4', 'C5']

        if plot == True:
            fig = plt.figure(figsize = (15,6))
            ax1 = fig.add_subplot(111)
            ax1.grid()
            strat_list = list(stats_combined.keys())
            strat_list.sort()
            cnum = 0
            for strat in strat_list:
                ax1.plot(stats_combined[strat]['CumulativePortfolioPerformance'], label = 'Strategy: ' + str(strat), color = colorlist[cnum])
                cnum += 1
            ax1.set_ylabel('Cumulative Returns [%]')
            fig.legend()
            fig.suptitle('Portfolio Performance with Beta Caps at -{} and {}'.format(beta_c, beta_c), fontsize=16, y = 1.05)


        fig2 = plt.figure(figsize = (18,12))
        
        shape1 = 3
        shape2 = 18
        colsp = int(shape2/shape1)
        rowsp = 1

        number_of_plots = len(stats_combined.keys())
        location_col_first_row = 0
        location_col_second_row = 3

        split_list = list(stats_combined.keys())
        split_list.sort()

        plot_number = 0
        for key in split_list:
            if plot_number < 3:
                plt.subplot2grid((shape1,shape2), (0,location_col_first_row), rowspan = rowsp, colspan=colsp, title = "Training Split: " + str(key))
                plt.plot(stats_combined[key]['CumulativePortfolioPerformance'], label = 'Strategy: '+ str(key), color = colorlist[plot_number])
                plt.plot(stats_combined[key]['CumulativeMarketPerformance'], label = 'Market for: ' + str(key), color = 'C0')
                plt.tight_layout()
                plt.grid()
                plt.legend()
                plt.tight_layout()
                plt.ylabel('Cumulative Returns [%]')
                location_col_first_row += 6
            else:
                plt.subplot2grid((shape1,shape2), (1,location_col_second_row), rowspan = rowsp, colspan=colsp, title = "Training Split: " + str(key))
                plt.plot(stats_combined[key]['CumulativePortfolioPerformance'], label = 'Strategy: '+ str(key), color = colorlist[plot_number])
                plt.plot(stats_combined[key]['CumulativeMarketPerformance'], label = 'Market for: ' + str(key), color = 'C0')
                plt.tight_layout()
                plt.grid()
                plt.legend()
                plt.ylabel('Cumulative Returns [%]')
                location_col_second_row += 6
            plot_number += 1
        plt.legend()


        results_cols_names = ['Annualized_Strategy_Return', 'Annualized_Market_Return', 
                              'Annualized_Strategy_Volatility','Annualized_Market_Volatility', 
                              'Annualized_Sharpe', 'Monthly_Sharpe','Max_Drawdown', 'Alpha', 'Treynor','Monthly_Sortino']
                
        combined_results = combined_results.T[['Annualized_Strategy_Return', 'Annualized_Market_Return', 
                              'Annualized_Strategy_Volatility','Annualized_Market_Volatility', 
                              'Annualized_Sharpe',  'Monthly_Sharpe', 'Max_Drawdown', 'Alpha', 'Treynor','Monthly_Sortino']]
        
        total_results_dictionary[key_beta] = combined_results.T    
    
    if beta_plot == True:
        
        if training_split_value_to_plot == 0:
            print("Need proper training split value...")
        else:
            fig3 = plt.figure(figsize = (15,6))
            ax3 = fig3.add_subplot(111)
            ax3.grid()
            for beta_v in beta_plot_dict.keys():
                ax3.plot(beta_plot_dict[beta_v]['CumulativePortfolioPerformance'], label = 'Beta Cap: ' + str(beta_v))
            ax3.plot(beta_plot_dict[beta_v]['CumulativeMarketPerformance'], label = 'Market')
            ax3.set_ylabel('Cumulative Returns [%]')
            fig3.legend()
            fig3.suptitle('Portfolio Performance with Varying Beta Caps at Training Size of: {}'.format(training_split_value_to_plot), fontsize=16, y = 1.05)


            
            
            
    return total_results_dictionary
    
    
    


# In[10]:


d = pd.read_pickle('results_by_training_cutoff_2.pkl')


total_dict = getStatsAll(d, beta_plot = True, training_split_value_to_plot=0.3)


# In[11]:


total_dict[1.0]


# In[ ]:





# In[23]:


def getStats(df, plot=False, arrow = False, advanced = False, random = False, risk_free_rate = 0.02):
    '''
    this function takes in a smaller subset of the initial raw data and computes only the metricw 
    '''
    
    global N
    
    if random == True:
        temp_mkt = np.random.randint(-10000,10000,len(df['pnl']))/100000
        temp_strategy = np.random.randint(-10000,10000,len(df['pnl']))/100000
        df['returns_strategy'] = temp_strategy
    
    cumulative_portfolio_performance = (df['pnl']-1000000)/1000000
    cumulative_market_performance = (1 + df['returns_mkt']).cumprod()-1
    
    if random == True:
        cumulative_portfolio_performance = (1 + df['returns_strategy'][1:]).cumprod()-1

    cumulative_sharpe = [np.NaN]
    cumulative_sortino = [np.NaN]
    cumulative_std = [np.NaN]
    cumulative_mean = [np.NaN]
    for date in df['returns_strategy'].index[1:]:
        cumulative_sharpe.append(sharpe(df['returns_strategy'][:date], risk_free_rate))
        cumulative_sortino.append(sortino(df['returns_strategy'][:date],risk_free_rate))
        cumulative_std.append(df['returns_strategy'][:date].std())

    cumulative_sortino = pd.Series(cumulative_sortino, index = df.index)
    cumulative_sharpe = pd.Series(cumulative_sharpe, index = df.index)
    cumulative_std = pd.Series(cumulative_std, index = df.index)
    
    sharpe_value = sharpe(df['returns_strategy'], risk_free_rate)
    sortino_value = sortino(df['returns_strategy'], risk_free_rate)
    
    max_drawdown, maxdrawdown_for_plotting = maxDD(df['pnl'])
    
    if random == True:
        temp_pnl = []
        count = 0
        for ret in df['returns_strategy']:
            count += 1
            if count == 1:
                temp_pnl.append(1000000*(1 + ret))
            else:
                temp_pnl.append(temp_pnl[-1] * (1 + ret))
        df['pnl'] = temp_pnl
        max_drawdown, maxdrawdown_for_plotting = maxDD(df['pnl'])


    
    portfolio_beta = portfolioBeta(df['returns_strategy'], df['returns_mkt'])
    treynor = (df['returns_strategy'].mean() - risk_free_rate)/portfolio_beta
    alpha = df['returns_strategy'].mean() - (risk_free_rate/12) - portfolio_beta * (df['returns_mkt'].mean() - (risk_free_rate/12))
    number_of_months = get_months(df.index[-1], df.index[0])
    
    annualized_strategy_return = (1 + df['returns_strategy']).prod()**(12/number_of_months) - 1
    annualized_market_return = (1 + df['returns_mkt']).prod()**(12/number_of_months) - 1
    
    monthly_strategy_volatility = df['returns_strategy'].std()
    monthly_market_volatility = df['returns_mkt'].std()


    sharpe_value = sharpe(df['returns_strategy'], risk_free_rate)
    sortino_value = sortino(df['returns_strategy'], risk_free_rate)
    
    
    annualized_strategy_volatility = monthly_strategy_volatility * np.sqrt(12)
    annualized_market_volatility = monthly_market_volatility * np.sqrt(12)
    
    annualized_sharpe_value = annualizedSharpe(annualized_strategy_return, annualized_strategy_volatility, risk_free_rate)
    
    stats_dict = {'Annualized_Strategy_Return':annualized_strategy_return,
                  'Annualized_Market_Return':annualized_market_return, 
                  'Annualized_Strategy_Volatility':annualized_strategy_volatility,
                  'Annualized_Market_Volatility':annualized_market_volatility,
                  'Annualized_Sharpe':annualized_sharpe_value,
                  'Monthly_Sharpe':sharpe_value, 'Monthly_Sortino':sortino_value, 
                  'Max_Drawdown':max_drawdown, 'Portfolio_Beta':portfolio_beta, 
                  'Alpha':alpha, 'Treynor':treynor}
        
        
    N += 1
    results = pd.DataFrame.from_dict(stats_dict, orient = "index", columns = ['Training Split: 0.3 '])
    
    
    maxdrawdown_for_plotting  = (maxdrawdown_for_plotting* cumulative_portfolio_performance).replace(0.0, np.NaN)
    index_to_start_arrow_at = maxdrawdown_for_plotting.idxmax(skipna = True)
    next_index = maxdrawdown_for_plotting.shift(1).idxmax(skipna = True)
    height_to_start_arrow_at = maxdrawdown_for_plotting.max()
    
    plot_minimum_limit = np.floor(min(min(cumulative_market_performance), min(cumulative_portfolio_performance)))
    plot_maximum_limit = np.ceil(max(max(cumulative_market_performance), max(cumulative_portfolio_performance)))
    
    if plot == True:
        
        fig = plt.figure(figsize = (15,6))
        ax1 = fig.add_subplot(111)
        ax1.plot(maxdrawdown_for_plotting, 'rv', ms = 8, label = 'MaxDrawdown')
        ax1.plot(cumulative_portfolio_performance, label = 'strategy', c = 'C0')
        ax1.plot(cumulative_market_performance, label = 'market', c = 'C1')

        if arrow == True:
            try:
                ax1.annotate('Max DD:{:6.2f}%'.format(max_drawdown*100), xy=(index_to_start_arrow_at, height_to_start_arrow_at),
                             xytext=(next_index, height_to_start_arrow_at),
                            arrowprops=dict(facecolor='black',arrowstyle="fancy", shrinkB = 5,
                                            connectionstyle = 'arc3, rad=-0.25'),
                             textcoords = 'data')
            except StopIteration:
                print('it is the arrows fault...')
                pdb.set_trace()

        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative Returns [%]')  
        ax1.set_ylim(plot_minimum_limit,plot_maximum_limit)
        ax1.grid()
        if advanced == True:
            plot_minimum_limit_ax2 = min(min(cumulative_sharpe), min(cumulative_sortino), min(cumulative_std))
            plot_maximum_limit_ax2 = max(max(cumulative_sharpe), max(cumulative_sortino), max(cumulative_std))
            
            ax2 = ax1.twinx()
            ax2.plot(cumulative_sharpe, linestyle = '--', label = 'cumulative sharpe', c = 'green')
            ax2.plot(cumulative_sortino, linestyle = '--', label = 'cumulative sortino', c = 'purple')
            ax2.plot(cumulative_std, linestyle = '--', label = 'cumulative standard dev.', c = 'black')
            ax2.set_ylabel('Sharpe, Sortino, Std. Dev.')
            ax2.set_ylim(plot_minimum_limit,plot_maximum_limit)

        fig.legend()
        fig.suptitle('Portfolio Performance with Beta Cap at -1 and 1', fontsize=16, y = 1.05)
    
#     print("----------------------------------------------------------")
#     print("|Annualized Strategy Return: HIGHER than MARKET is BETTER|")
#     print("|Sharpe Ratio: HIGHER is BETTER                          |")
#     print("|Sortino Ratio: HIGHER is BETTER                         |")
#     print("|Max Drawdown: LOWER is BETTER                           |")
#     print("|Alpha: HIGHER is BETTER                                 |")
#     print("|Treynor: HIGHER is BETTER                               |")
#     print("----------------------------------------------------------")
    print(cumulative_portfolio_performance[-1], cumulative_market_performance[-1])
    return results.T


