#!/usr/bin/env python
# coding: utf-8

# # File Declaration:
# The purpose of this file is to create a single data structure containing our universe of tickers, each with sentiment data, selected fundamental data, and end of day data. Our overall universe contains 1130 individual tickers. 

# In[156]:


# ### Necessary Packages
# - **Numpy**: was not used...may delete later
# - **Pandas**: primarily used for dataframe structures
# - **pdb**: used for debugging
# - **pickle**: used for porting non-dataframe structures into files
# - **quandl**: used for querying the quandl database
# - **Counter**: easy way to count unique items in a list and return a simple dictionary
# - **time**: used for creating timers
# - **clear_output**: used for managing output organization
# - **os, listdir, isfile, and join**: used for operating system path manipulation

# In[157]:


import numpy as np
import pandas as pd
import pdb
import pickle
import quandl
from collections import Counter
import time
pd.set_option("display.max_columns", 101)
from IPython.display import clear_output

import os
from os import listdir
from os.path import isfile, join


# #### Quandl Key: Insert quandl key here

# In[158]:


quandl.ApiConfig.api_key = BenjaminMorganPrivateKeys['Quandl']


# #### Function Definitions:
# - **clean_period_type**: for cleaning the fundamental data and only keeping quarterly.
# - **adjust_data_by_years**: for adjusting fundamental data early on to reduce size of data structure that would be manipulated in the code.
# - **get_fundamental_data**: this function finds all files in the directory, lists them, then calls two functions to keep only certain file types and then merge the files together. Then, it returns the fundamental data. 

# In[159]:


def clean_period_type(df):
    '''
    Returns only the quarterly data in the original dataframe.  
    '''
    
    df = df[df['PER_TYPE'] == 'Q']
    return df

def adjust_data_by_years(df, start_year, end_year):
    '''
    Adds a way to truncate the data based on a start and end year. 
    '''
    
    df = df[str(start_year):str(end_year)]    
    return df

def get_fundamental_data():
    '''
    Get path to local directory. Find Zacks files and combine them into one data structure. 
    '''
    
    cwd = os.getcwd()
#     file_names = !ls
    file_names = [f for f in listdir(cwd) if isfile(join(cwd, f))] 
    file_names = list(file_names)
    files = keep_only_certain_file_types(file_names, 'csv')
    fundamental_data = merge_files(files)
    return fundamental_data


# #### Function Definitions:
# - **keep_only_certain_file_types**: this function keeps only certain file types
# - **merge_files**: merges the list of files and returns a total_data variable with all the fundamental data in the structure. 

# In[208]:


def keep_only_certain_file_types(files_list, file_type):
    '''
    take in the files from the directory, take in the file you want to keep, return the appropriate files. 
    '''
    dictionary = {}
    for file in files_list:
        try:
            dictionary[file.split('.')[0]] = file.split('.')[1]
        except IndexError:
            pass
    df = pd.DataFrame.from_dict(dictionary, orient='index', columns=['File_Type'])
    for index, row in df.iterrows():
        if row['File_Type'] != file_type:
            df.drop(index, axis = 0, inplace = True)
    
    dictionary = df['File_Type'].to_dict()        
    list_of_files = []
    for k, v in dictionary.items():
        list_of_files.append('.'.join([k,v]))
    
    file_list = []
    for file in list_of_files:
        if file[0] != 'Z':
            pass
        else:
            file_list.append(file)
    
    return file_list

def merge_files(list_of_files):
    '''
    Merging of the dataframes.
    '''
    total_data = pd.DataFrame()
    for file in list_of_files:
        data_file = pd.read_csv(file, sep = '|')
        total_data = total_data.append(data_file, ignore_index = True)
    return total_data


# #### Function Definitions:
# - **get_fundamental_columns**: is a simple way to obtain the necessary fundamental columns from the fundamental data. This is not a delete function but a replace and is therefore easily adaptable as long as the column is from the original 275 columns. 

# In[161]:


def get_fundamental_columns():
    
    '''
    list of fundamental columns to keep. Can add from quandl's Zacks fundamental data. 
    '''
    
    fund_cols_to_keep = [
        'TICKER',
        'WAVG_SHARES_OUT',
        'CURR_RATIO',
        'GROSS_MARGIN',
        'OPER_PROFIT_MARGIN',
        'EBIT_MARGIN',
        'PRETAX_PROFIT_MARGIN',
        'PROFIT_MARGIN',
        'FREE_CASH_FLOW',
        'ASSET_TURN',
        'INVTY_TURN',
        'RCV_TURN',
        'DAY_SALE_RCV',
        'RET_EQUITY',
        'RET_TANG_EQUITY',
        'RET_ASSET',
        'RET_INVST',
        'FREE_CASH_FLOW_PER_SHARE',
        'BOOK_VAL_PER_SHARE',
        'OPER_CASH_FLOW_PER_SHARE'
    ]
    
    return fund_cols_to_keep


# #### Function Definitions:
# - **filter_and_adjust_fundamental_data**: takes in the fundamental dataframe, a start date, and an end date and filters the dataframe and then truncates the dataframe to only include the data between the dates. 

# In[162]:


def filter_and_adjust_fundamental_data(fd, s_date_var, e_date_var):
    '''
    more dataframe handling. 
    '''
    fd_copy = fd.copy()
    fd_copy.set_index('PER_END_DATE', inplace = True)
    fd_copy.index = pd.to_datetime(fd_copy.index)
    fd_copy = clean_period_type(fd_copy)
    fd_copy.sort_index(inplace = True)
    fd_copy = adjust_data_by_years(fd_copy, s_date_var, e_date_var)
    fund_columns = get_fundamental_columns()
    fd_copy = fd_copy[fund_columns]
    
    return fd_copy


# #### Function Definitions:
# - **separate_sentiment_data**: takes in a dataframe of the sentiment data and a list of tickers pulled from the fundamental data and separates the dataframe into a dictionary where the keys are the tickers and the dataframe consists of the sentiment columns.
# - **append_NS1**: takes the ticker and adds the database to the front of the ticker and adds the country code to the end of ticker. Also handles a . that is sometimes at the end of the ticker name by adding a 'ZZZZ' character to the string. 

# In[163]:


def separate_sentiment_data(sda, avail_ticks):
    '''
    takes in a dataframe of the sentiment data and a list of tickers pulled from the fundamental data 
    and separates the dataframe into a dictionary where the keys are the tickers 
    and the dataframe consists of the sentiment columns.
    '''
    all_sentiment_dictionary = {}
    for ticker in avail_ticks:
        temp_dfs = pd.DataFrame(index = sda.index)
        for col in sda.columns:
            if ticker in col:
                if 'Not Found' in col.split(' - ')[1]:
                    pass
                elif sda[col][-5:].isna().sum() > 4:
                    pass
                elif sda[col].empty == True:
                    pass
                elif len(sda[col]) < 500:
                    pass
                elif col[:3] == 'NS1':
                    temp_dfs[col.split(' - ')[1].replace(' ', '_')] = sda[col]
            if temp_dfs.empty == True:
                pass
            else:
                all_sentiment_dictionary[ticker] = temp_dfs
    return all_sentiment_dictionary

def append_NS1(lt):
    '''
    takes the ticker and adds the database to the front of the ticker and adds the country code to the end of ticker. 
    Also handles a . that is sometimes at the end of the ticker name by adding a 'ZZZZ' character to the string. 
    '''
    if lt[-1] == '.':
        lt = lt.replace('.', '')
        temp = 'NS1/' + lt + 'ZZZZ_US'
    else:
        lt = lt.replace('.', '_')
        temp = 'NS1/' + lt + '_US'
    return temp


# #### Function Definitions:
# - **separate_eod_data_into_dictionary**: takes in the raw single-dataframe end of day data and a ticker list and then separates the tickers into keys of a dictionary and populates the values as dataframes with the end of day data, which is adjusted close and adjusted volume. 
# - **append_eod**: takes a ticker and appends the EOD database code with the necessary numerical value that corresponds to the columns of the end of day data. 

# In[164]:


def separate_eod_data_into_dictionary(raw_eod_data, tickers):
    '''
    Take the end of day dataframe of all tickers and separate it in to a dictionary with tickers as keys and 
    dataframes of EOD data as values.
    '''
    
    all_eod_data_copy = raw_eod_data.copy()
    all_eod_dictionary = {}
    for ticker in tickers:
        temp_df = pd.DataFrame(index = all_eod_data_copy.index)
        for col in all_eod_data_copy.columns:            
            if ticker in col:
                if 'Not Found' in col.split(' - ')[1]:
                    pass
                elif all_eod_data_copy[col][-5:].isna().sum() > 4:
                    pass
                elif all_eod_data_copy[col].empty == True:
                    pass
                elif len(all_eod_data_copy[col]) < 500:
                    pass
                elif col[:3] == 'EOD':
                    temp_df[col.split(' - ')[1]] = all_eod_data_copy[col]
        if temp_df.empty == True:
            pass
        else:
            all_eod_dictionary[ticker] = temp_df
    return all_eod_dictionary

def append_eod(lt):
    
    '''
    takes a ticker and appends the EOD database code with the necessary numerical value that 
    corresponds to the columns of the end of day data. 
    '''
    
    if lt[-1] == '.':
        lt = lt.replace('.', '')
        temp = 'EOD/' + lt + 'ZZZ.11'
        temp2 = 'EOD/' + lt + 'ZZZ.12'
    else:
        lt = lt.replace('.', '_')
        temp = 'EOD/' + lt + '.11'
        temp2 = 'EOD/' + lt + '.12'
    return tuple([temp, temp2])


# #### Function Definitions:
# - **create_master_dictionary**: creates a master dictionary with two keys, raw_data and factors. The factors key is populated with a dictionary of monthly factors and daily factors using fama-french five factor cvs files. The raw data value is the dictionary of all the tickers as keys and each ticker's respective dataframe consisting of the fundamental, sentiment, and end of day columns. 
# - **clean_up_nonexistent_data**: takes in a data dictionary and for each key/ticker in the dictionary, will check for any nans in the dataframe and delete if any are present. 
# - **check_for_nans**: takes in a dictionary and counts nans. 
# - **write_to_pickle**: takes a file and file name and writes it to a pickle file in the local directory. 
# - **get_QQQ**: returns the end of day data for the market, using NASDAQ/QQQ as the market. 

# In[165]:


def create_master_dictionary(merged_all_data):
    
    '''
    Creates a master dictionary with two keys, raw_data and factors. 
    The factors key is populated with a dictionary of monthly factors 
    and daily factors using fama-french five factor cvs files. 
    The raw data value is the dictionary of all the tickers as keys and 
    each ticker's respective dataframe consisting of the fundamental, 
    sentiment, and end of day columns.
    '''
    
    monthly_factors = pd.read_csv('ff_five_factors_monthly.csv', index_col = 0)
    daily_factors = pd.read_csv('ff_five_factors_daily.csv', index_col = 0)
    monthly_factors.index = pd.to_datetime(monthly_factors.index)
    daily_factors.index = pd.to_datetime(daily_factors.index)

    factors = {'Monthly_Factors':monthly_factors, 'Daily_Factors':daily_factors}

    all_data = {}
    all_data['Raw_Data'] = merged_all_data
    all_data['Factors'] = factors
    
    return all_data

def clean_up_nonexistent_data(data_dict):
    '''
    takes in a data dictionary and for each key/ticker in the dictionary, 
    will check for any nans in the dataframe and delete if any are present
    '''
    temp_dict = {}
    for key in data_dict.keys():
        if data_dict[key][data_dict[key].columns[0]].isna().sum() > 0:
            pass
        else:
            temp_dict[key] = data_dict[key]
    return temp_dict

def check_for_nans(t_d_c):
    '''
    takes in a dictionary and counts nans in entire dictionary of dataframes
    '''
    sums = 0
    for key in t_d_c.keys():
        for col in t_d_c[key]:
            sums += t_d_c[key][col].isna().sum()
    return sums

def write_to_pickle(data_file_to_write, fn):
    '''
    takes a file and file name and writes it to a pickle file in the local directory.
    '''
    with open(fn + '.pickle', 'wb') as handle:
        pickle.dump(data_file_to_write, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return None

def get_QQQ():
    
    '''
    returns the end of day data for the market, using NASDAQ/QQQ as the market.
    '''
    
    start_date_var = '2013-01-01'
    start_date_fund = '2012-01-01'
    end_date_var = '2019-12-31'
    end_date_fund = '2019-12-31'
    
    QQQ = pd.DataFrame(quandl.get('EOD/QQQ', start_date = start_date_fund, end_date = end_date_fund)['Adj_Close'])
    QQQ.index = pd.to_datetime(QQQ.index)
    QQQ.index.rename('date', inplace = True)
    QQQ.rename(columns = {'Adj_Close': 'QQQ_Adj_Close'}, inplace = True)
    return QQQ


# #### Function Definitions: 
# - **concat_dicts**: takes 3 specific data structures of fundamental data, sentiment data, and end of day data and then concatenates everything in a dataframe on a datetime index by ticker. It takes the ticker and stores it as the key in a temporary dictionary and then takes the associated dataframe and stores it as a value. 

# In[166]:


def concat_dicts(fundamental_data_dictionary, sentiment_data_dictionary, end_of_day_data_dictionary):
    '''
    takes 3 specific data structures of fundamental data, sentiment data, and end of day data 
    and then concatenates everything in a dataframe on a datetime index by ticker. 
    It takes the ticker and stores it as the key in a temporary dictionary and then takes the 
    associated dataframe and stores it as a value.
    '''
    
    clear_output()
    temp_data_concat = {}
    t0 = time.time()
    count = 0
    count_concat = 0
    
    QQQ = get_QQQ()
    selected_tickers = list(end_of_day_data_dictionary.keys())
    for ticker in selected_tickers:
        count += 1
        try:
            e_data = end_of_day_data_dictionary[ticker]
            s_data = sentiment_data_dictionary[ticker]
            f_data = fundamental_data_dictionary[fundamental_data_dictionary['TICKER'] == ticker]

            if e_data.index[0].year >= 2014:
                pass
            elif f_data[-3:].isna().sum().sum() > 3:
                pass
            elif f_data[0:3].isna().sum().sum() >= 1:
                pass
            else:
                count_concat += 1
                e_data.sort_index(inplace = True)
                e_data.index.rename('date', inplace = True)

                f_data.sort_index(inplace = True)
                f_data.index.rename('date', inplace = True)

                temp_data_concat[ticker] = pd.concat([e_data, f_data, s_data, QQQ], axis = 1)
                temp_data_concat[ticker].sort_index(inplace = True)
                temp_data_concat[ticker].fillna(method = 'ffill', inplace = True)
                temp_data_concat[ticker] = temp_data_concat[ticker]['2013-01-02':]
                temp_data_concat[ticker].drop(['TICKER'], axis = 1, inplace = True)
            if count%100 == 0: 
                clear_output()
                print("Selecting..."+ str(count_concat) + " tickers out of " + str(count) + ' in '  + str(round(time.time()-t0, 2)) + " seconds...")


        except KeyError:
            pass
    t1 = time.time()
    clear_output()
    print("Selected..."+ str(count_concat) + " tickers out of " + str(count) + ' in '  + str(round(t1-t0, 2)) + " seconds...")
    # print("Tickers for Selection: " + str(count_concat) + " & Total Tickers Searched: " + str(count) + " & Time Elapsed: " + str(round(t1-t0, 2)) + " seconds...")
    print("Finished Selecting...")
    
    number_of_nans = check_for_nans(temp_data_concat)
    if number_of_nans > 0:
        print("Error with eliminating NaNs...")
        print("NaNs found: ", number_of_nans)
    else:
        pass
    
    complete_dataset = create_master_dictionary(temp_data_concat)
    fn = 'complete_dataset'
    write_to_pickle(complete_dataset, fn)
    clear_output()
    print('Data compilation is complete. ')
    print('' + str(count_concat) +' tickers in universe.')
    print('File name is ' + str(fn) + '.pickle and is located in the local directory.')
    return complete_dataset


# #### Function Definitions:
# - **getFundamentalData**: gets the fundamental data using the above functions or quandl. 
# - **getSentimentData**: gets the sentiment data using the above functions or quandl. 
# - **getEndOfDayData**: gets the end of day data using the above functions or quandl. 
# 

# In[167]:


def getFundamentalData():
    
    '''
    gets the fundamental data using provided functions, filters and adjusts the data, 
    and then gets a list of available tickers from the 
    '''
    
    clear_output()
    start_date_var = '2013-01-01'
    start_date_fund = '2012-01-01'
    end_date_var = '2019-12-31'
    end_date_fund = '2019-12-31'
    
    fundamental_data = get_fundamental_data()
    fundamental_data_copy = filter_and_adjust_fundamental_data(fundamental_data, start_date_fund, end_date_fund)
    internal_list_of_tickers_from_fundamental = list(Counter(fundamental_data_copy['TICKER']).keys())
    
    return (fundamental_data_copy, internal_list_of_tickers_from_fundamental)

def getSentimentData(list_of_tickers_from_fundamental):
    '''
    gathers sentimental data from quandl, or a local pickle file from a previous compilation.
    '''
    clear_output()
    available_tickers = list_of_tickers_from_fundamental
    available_tickers_NS1 = pd.Series(available_tickers).apply(append_NS1)
    available_tickers_NS1 = available_tickers_NS1.tolist()
    
    #get sentiment data
    # sentiment_data_all = quandl.get(available_tickers_NS1, start_date = start_date_fund, end_date = end_date_fund)
    # sentiment_data_all.to_csv('raw_all_sentiment.csv')
    sentiment_data_all = pd.read_pickle('raw_all_sentiment.pickle')
    all_sentiment_dictionary = separate_sentiment_data(sentiment_data_all, available_tickers)
    internal_list_of_tickers_from_sentiment = list(all_sentiment_dictionary.keys())
    
    return (all_sentiment_dictionary, internal_list_of_tickers_from_sentiment)

def getEndOfDayData(list_of_tickers_from_fundamental, list_of_tickers_from_sentiment):
    
    '''
    gathers EOD data by first finding minimum list of tickers from fundamental and sentiment and 
    then getting EOD data from quandl or in this case, a precompiled file of end of day data. 
    '''
    
    clear_output()
    if len(list_of_tickers_from_fundamental) >= len(list_of_tickers_from_sentiment):
        ticker_list = list_of_tickers_from_sentiment
    else:
        ticker_list = list_of_tickers_from_fundamental
    
    ticker_series = pd.Series(ticker_list).apply(append_eod)
    ticker_list_adjusted = [i for sub in ticker_series for i in sub]
    
    #get eod data
    #print("Getting EOD data from quandl...")
    # all_eod_data = quandl.get(ticker_list_adjusted, start_date = start_date_var, end_date = end_date_var)

    #read eod data
    print("Reading EOD data from pickle file...")
    all_eod_data = pd.read_pickle('sentiment_tickers_eod_data.pickle')
    all_eod_dictionary = separate_eod_data_into_dictionary(all_eod_data, ticker_list)
    cleaned_all_eod_dictionary = clean_up_nonexistent_data(all_eod_dictionary)
    
    internal_list_of_tickers_from_endofday = list(cleaned_all_eod_dictionary.keys())
    
    return (cleaned_all_eod_dictionary, internal_list_of_tickers_from_endofday)


# #### Function Definitions:
# - **start_running**: runs the three necessary functions to get the data and then the concat_dicts function to put them all together and store that as a pickle file in the local directory. 

# In[170]:


def start_running():
    '''
    runs the necessary functions to get the different data, sentiment, eod, and fundamental 
    and then concatenates them all together and stores that as a pickle file in the local directory
    '''
    print('Expected time to run: 20 minutes.')
#     tstart = time.time()
    fundamental_dataset, fundamental_dataset_tickers = getFundamentalData()
    sentiment_dataset, sentiment_dataset_tickers = getSentimentData(fundamental_dataset_tickers)
    endofday_dataset, endofday_dataset_tickers = getEndOfDayData(fundamental_dataset_tickers, sentiment_dataset_tickers)
    c_dict = concat_dicts(fundamental_dataset, sentiment_dataset, endofday_dataset)
    return c_dict
    

