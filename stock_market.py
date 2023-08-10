import numpy as np
import pandas as pd
import os


parent_dir = "C:\Dennis\Personal\Projects\ml_playground\data\stock_market"
stocks_dir = "C:\Dennis\Personal\Projects\ml_playground\data\stock_market\stocks"
etfs_dir = "C:\Dennis\Personal\Projects\ml_playground\data\stock_market\etfs"

stock_list = os.listdir(stocks_dir)
etfs_list = os.listdir(etfs_dir)