import os
import numpy as np
from PIL import Image
import tools

path = 'C:\Dennis\Personal\Projects\ml_playground\data\credit_score\\application_record.csv'


df = tools.import_data(path)

usable_cols = df.select_dtypes(include=int)

print(usable_cols.head())
print(usable_cols['CNT_CHILDREN'].unique())