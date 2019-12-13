import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt

plt.style.use('seaborn')
import seaborn as sns

import re
import xgboost as xgb

import warnings

warnings.filterwarnings("ignore")

df_initial = pd.read_csv('/Users/joelnorthrup/Desktop/listings_summary.csv')

# checking shape
print("The dataset has {} rows and {} columns.".format(*df_initial.shape))

# ... and duplicates
print("It contains {} duplicates.".format(df_initial.duplicated().sum()))

df_initial.head(1)

df_initial.columns



print(50 * '=')
print('Section: Exploring the Airbnb Berlin dataset')
print(50 * '-')


print('Dataset excerpt:\n\n', df_initial.head())

new_df = df_initial.drop(['summary', 'space', 'description', 'neighborhood_overview','notes', 'transit', 'access', 'interaction','house_rules', 'thumbnail_url', 'host_about', 'license',
'listing_url', 'scrape_id', 'last_scraped', 'name','medium_url', 'picture_url', 'xl_picture_url', 'host_id', 'host_since','host_location', 'host_acceptance_rate', 'host_neighbourhood',
'host_listings_count', 'host_total_listings_count','host_url', 'host_name', 'host_thumbnail_url', 'host_picture_url','neighbourhood', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed',
'city', 'state', 'street', 'zipcode', 'market', 'smart_location', 'country_code','country', 'is_location_exact','weekly_price', 'monthly_price', 'square_feet',
'minimum_nights', 'maximum_nights', 'calendar_updated', 'calendar_last_scraped', 'first_review', 'last_review','jurisdiction_names', 'experiences_offered','host_verifications'],axis=1)


half =len(new_df.index)/2
half



'minimum_minimum_nights', 'maximum_minimum_nights','minimum_maximum_nights', 'maximum_maximum_nights','number_of_reviews_ltm',


print(50 * '=')
print('Section: Visualizing the important characteristics of a dataset')
print(50 * '-')

sns.set(style='whitegrid', context='notebook')
#cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']

sns.pairplot(new_df, size=2.5)
# plt.tight_layout()
# plt.savefig('./figures/scatter.png', dpi=300)
plt.show()


cm = np.corrcoef(new_df.values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15},
                 yticklabels=new_df,
                 xticklabels=new_df)

# plt.tight_layout()
# plt.savefig('./figures/corr_mat.png', dpi=300)
plt.show()

sns.reset_orig()

# x axis values
x = [1, 2, 3]
# corresponding y axis values
y = [2, 4, 1]

# plotting the points
plt.plot(x, y)

# naming the x axis
plt.xlabel('x - axis')
# naming the y axis
plt.ylabel('y - axis')

# giving a title to my graph
plt.title('My first graph!')

# function to show the plot
plt.show()






def dropCols(self, toDrop=[
    'summary', 'space', 'description', 'neighborhood_overview',
    'notes', 'transit', 'access', 'interaction',
    'house_rules', 'thumbnail_url', 'host_about', 'license',
    'listing_url', 'scrape_id', 'last_scraped', 'name',
    'medium_url', 'picture_url', 'xl_picture_url', 'host_id', 'host_since',
    'host_location', 'host_acceptance_rate', 'host_neighbourhood',
    'host_listings_count', 'host_total_listings_count',
    'host_url', 'host_name', 'host_thumbnail_url', 'host_picture_url',
    'neighbourhood', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed',
    'city', 'state', 'street', 'zipcode', 'market', 'smart_location', 'country_code',
    'country', 'is_location_exact',
    'weekly_price', 'monthly_price', 'square_feet',
    'minimum_nights', 'minimum_minimum_nights', 'maximum_minimum_nights',
    'maximum_nights', 'minimum_maximum_nights', 'maximum_maximum_nights',
    'calendar_updated', 'calendar_last_scraped',
    'number_of_reviews_ltm', 'first_review', 'last_review',
    'jurisdiction_names', 'experiences_offered',
    'host_verifications']):
    self.df = self.df.drop(toDrop, axis=1)
    self.columns = self.df.columns.values.tolist()


