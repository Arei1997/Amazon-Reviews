import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False) 
# setting the style of the notebook to be monokai theme  
# this line of code is important to ensure that we are able to see the x and y axes clearly
# If you don't run this code line, you will notice that the xlabel and ylabel on any plot is black on black and it will be hard to see them. 


# Load the data
reviews_df = pd.read_csv('amazon_reviews.csv')
reviews_df
print(reviews_df)

# View the DataFrame Information
reviews_df.info()

# View DataFrame Statistical Summary
reviews_df.describe()
# View DataFrame Statistical Summary
print(reviews_df.describe())


# Dropping the date column
reviews_df.drop(columns=['date'], inplace=True)
print(reviews_df)

# check to verify colum was dropped
if 'date' not in reviews_df:
    print("Column 'Column_Name' was successfully dropped.")
else:
    print("Column 'Column_Name' was not dropped.")