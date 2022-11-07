#!/usr/bin/env python
# coding: utf-8

# ## Import Packages

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from pandas.api.types import CategoricalDtype

sns.set_theme(font_scale=1.5, style="darkgrid")


# ## Read Data

# In[3]:


# Read indivual scripts
sd_listings = pd.read_csv('./Data/Listings/san_diego_listings.csv')
oakland_listings = pd.read_csv('./Data/Listings/oakland_listings.csv')
la_listings = pd.read_csv('./Data/Listings/los_angeles_listings.csv')
sf_listings = pd.read_csv('./Data/Listings/san_francisco_listings.csv')
scc_listings = pd.read_csv('./Data/Listings/santa_clara_county_listings.csv')
smc_listings = pd.read_csv('./Data/Listings/san_mateo_county_listings.csv')
sc_listings = pd.read_csv('./Data/Listings/santa_cruz_county_listings.csv')
pg_listings = pd.read_csv('./Data/Listings/pacific_grove_listings.csv')

#listings are split by states, so we should combine them while keeping
listings = [sd_listings, oakland_listings, la_listings, sf_listings, scc_listings, smc_listings, sc_listings, pg_listings]
states = ['San Diego', 'Oakland', 'Los Angeles', 'San Francisco', 'Santa Clara County', 'San Mateo County', 'Santa Cruz County', 'Pacific Grove']

for i in range(len(listings)):
  listings[i]['state'] = states[i]

all_listings = pd.concat(listings, axis=0)
all_listings.head()


# ### Drop some columns

# In[4]:


all_listings.info()


# With a total of 75 columns, which is too many for a model, we had to remove columns. Initially, we
# removed those columns which are not useful for predicting price (e.g. url, host-related features that are unrelated to the property, etc).

# In[5]:


all_listings.drop(['listing_url', 'scrape_id', 'last_scraped', 'source', 'picture_url', 'host_id', 
               'host_url', 'host_thumbnail_url', 'host_picture_url','calendar_updated', 
               'calendar_last_scraped', 'calculated_host_listings_count','calculated_host_listings_count_entire_homes', 
               'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms',
               'host_since','host_location','host_about','host_neighbourhood','host_listings_count',
               'host_total_listings_count','host_total_listings_count','host_verifications'], axis=1, inplace=True)


# #### Remove columns with majority null values

# In[6]:


(all_listings.isna().sum() / all_listings.isna().count()) *100


# From here, it seems like we can drop neighborhood_overview, neighbourhood, neighbourhood_group_cleansed, bathrooms, license. 
# 
# However, we will keep neighborhood_overview to potentially get it's sentiment value.

# In[7]:


all_listings['license'].value_counts()


# It can be that those who have a null value in the license column just don't have a license. Maybe we can try for a boolean category here.

# In[8]:


all_listings['has_license'] = all_listings['license'].notnull()*1
all_listings.drop("license", axis = 1, inplace= True)


# Let's remove the rest

# In[9]:


all_listings.drop("neighbourhood", axis = 1, inplace= True)
all_listings.drop("neighbourhood_group_cleansed", axis = 1, inplace= True)
all_listings.drop("bathrooms", axis = 1, inplace= True)


# Remove columns that have 1 or 0 unique values

# In[10]:


# get number of unique values for each column
counts = all_listings.nunique()
counts_dict = counts.to_dict()

# record columns to delete (those that only have 1 or 0 unique values)
to_del = [i for i,v in counts_dict.items() if v == 1 or v == 0]
print(to_del)

# drop those columns
all_listings.drop(to_del, axis=1, inplace=True)
print(all_listings.shape)


# There are multiple columns for minimum and maximum night stays which seem to have minimal differences. The default min/max night stay values will be used instead.

# In[11]:


all_listings.drop(['minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights', 'maximum_maximum_nights', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm'], axis=1, inplace=True)


# ### Boolean columns

# In[12]:


# Replacing columns with f/t with 0/1
all_listings.replace({'f': 0, 't': 1}, inplace=True)

# Plotting the distribution of numerical and boolean categories
all_listings.hist(figsize=(20,20))


# Drop those that have mostly just one category

# In[13]:


all_listings.drop(['has_availability', 'host_has_profile_pic','has_availability'], axis=1, inplace=True)


# #### Name

# In[14]:


all_listings['name'].isna().sum()


# There's only 3 null values, we can just remove those rows.

# In[15]:


all_listings = all_listings[all_listings['name'].notna()]


# #### host response time

# In[16]:


print("Null values:", all_listings.host_response_time.isna().sum())
print(f"Proportion: {round((all_listings.host_response_time.isna().sum()/len(all_listings))*100, 1)}%")

# Number of rows without a value for host_response_time which have also not yet had a review
len(all_listings[all_listings.loc[ :,['host_response_time', 'first_review'] ].isnull().sum(axis=1) == 2])


# We'll fill the NA values wioth its own category since it's a signficant proportion of the data

# In[17]:


all_listings.host_response_time.fillna("NA", inplace=True)
all_listings.host_response_time.value_counts(normalize=True)


# #### host response rate

# In[18]:


print("Null values:", all_listings.host_response_rate.isna().sum())
print(f"Proportion: {round((all_listings.host_response_rate.isna().sum()/len(all_listings))*100, 1)}%")


# In[19]:


# Removing the % sign from the host_response_rate string and converting to an integer
all_listings.host_response_rate = all_listings.host_response_rate.str[:-1].astype('float')

print("Mean host response rate:", round(all_listings['host_response_rate'].mean(),0))
print("Median host response rate:", all_listings['host_response_rate'].median())
print(f"Proportion of 100% host response rates: {round(((all_listings.host_response_rate == 100.0).sum()/all_listings.host_response_rate.count())*100,1)}%")


# In[20]:


all_listings.host_response_rate.value_counts()


# The distribution is right skewed, so we can create categories of this instead.

# In[21]:


bins = [0, 50, 70, 80, 90, 95, 101]
response_rate_labels = ['unresponsive', 'somewhat unresponsive', 'somewhat responsive', 'responsive', 'very responsive', 'extremely responsive']
all_listings['host_response_rate'] = pd.cut(all_listings['host_response_rate'], bins=bins, labels=response_rate_labels, right=False)


# In[22]:


cat_dtype_response = CategoricalDtype(categories=response_rate_labels+['NA'])
all_listings['host_response_rate'] = all_listings['host_response_rate'].astype(cat_dtype_response)
all_listings['host_response_rate'] = all_listings['host_response_rate'].fillna('NA')


# #### Property Type

# In[23]:


all_listings.property_type.value_counts()


# There's a lot of categories here which may cause problems in the model. We can group them into specific categories that roughly summarise each property_type. We can notice houses tend to use the term "Entire", while shared and private rooms are explicitly stated.

# In[24]:


property_categorisation = {
    'Entire': 'House',
    'Private room': 'Private Room',
    'Shared room': 'Shared Room',
    'Tiny home': 'House',
    'Room': 'Private Room'
}

def map_property_type(property_type):
    for key,value in property_categorisation.items():
        if key in property_type:
            return value
    return 'Others'

all_listings['property_type'] = all_listings['property_type'].apply(map_property_type)


# In[25]:


all_listings['property_type'].value_counts()


# ### Price

# The price column is formatting as a string with symbols. We'll fix that and remove NA and 0 values.

# In[26]:


all_listings["price"] = all_listings["price"].str[1:].str.replace(",","").astype("float")


# In[27]:


all_listings = all_listings[all_listings['price'].notna()] 
all_listings = all_listings[all_listings.price!=0]


# ### Bathrooms

# It's better to have separate columns for the number of bathrooms and the type of bathroom

# In[28]:


all_listings["bathrooms"] = all_listings["bathrooms_text"].str.split(" ", expand=True)[0]
all_listings["bathroom_type"] = all_listings["bathrooms_text"].str.split(" ", expand=True)[1]
all_listings.drop("bathrooms_text", axis = 1, inplace= True)


# In[29]:


all_listings['bathrooms'] = all_listings['bathrooms'].fillna('NA')
all_listings['bathroom_type'] = all_listings['bathroom_type'].fillna('NA')


# Bathrooms should ideally be a count. Some of them use terms to describe the bathroom, so we will map them to numbers.

# In[30]:


mapping = {'Private': '1', 'Shared': '1', 'Half-bath': '1', 'NA': '1'}

all_listings = all_listings.replace({'bathrooms': mapping})

all_listings['bathrooms'] = all_listings['bathrooms'].astype('str').astype('float')


# ### Bedrooms & Beds

# There's a portion of null values for bedrooms and beds. We will derive a quick linear regression model to estimate the missing values

# In[31]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

bedroom_not_na = all_listings[all_listings['bedrooms'].notna()]
bedrooms = bedroom_not_na['bedrooms']
accommodates = bedroom_not_na[['accommodates']]
bedrooms_model = LinearRegression()
bedrooms_model.fit(accommodates, bedrooms)
bedroom_pred = bedrooms_model.predict(accommodates)
mean_squared_error(bedrooms, bedroom_pred)


# In[32]:


bed_not_na = all_listings[all_listings['beds'].notna()]
beds = bed_not_na['beds']
accommodates = bed_not_na[['accommodates']]
beds_model = LinearRegression()
beds_model.fit(accommodates, beds)
bedroom_pred = beds_model.predict(accommodates)
mean_squared_error(beds, bedroom_pred)


# The MSE for both are relatively low, so we can use these models to fill up null values

# In[33]:


import math
import warnings

warnings.simplefilter("ignore", UserWarning)
for index, row in all_listings.iterrows():
  value = row['accommodates']
  if math.isnan(row['bedrooms']):
    all_listings.at[index, 'bedrooms'] = round(bedrooms_model.predict([[value]])[0])
  if math.isnan(row['beds']):
    all_listings.at[index, 'beds'] = round(beds_model.predict([[value]])[0])


# ### Amenities

# This column has a list of amenities which are free text for users (so it will be a problem when using it for a model). 
# 
# The first approach is to simply take a count of the amenities, as we can assume more amenities provided translates to a higher price. 
# 
# Another way would be to group the amenities into specific groups based on certain keywords.

# In[34]:


import ast

def get_length(text):
  amenities_list = ast.literal_eval(text)
  return len(amenities_list)

all_listings['num_of_amenities'] = all_listings.amenities.apply(get_length)
fig, ax = plt.subplots(figsize=(12, 6))
hp = sns.histplot(all_listings['num_of_amenities'], bins=25)


# In[35]:


import ast

all_listings.reset_index(inplace=True)
amenities_categorisation = {
    'essentials': ['soap', 'shampoo', 'towel', 'conditioner', 'toiletries', 'linen', 'water'],
    'luxury': ['pool', 'tub', 'park', 'sound', 'wifi'],
    'appliances': ['refrigerator', 'stove', 'oven', 'washer', 'dryer', 'microwave'],
    'comfort': ['heat', 'air condition'],
    'entertainment': ['tv', 'console', 'gym', 'game', 'entertainment'],
    'security': ['lock', 'alarm', 'guard'],
    'furniture': ['storage', 'chair', 'table', 'bed']
}

for category in amenities_categorisation.keys():
  all_listings[category] = 0
all_listings['miscellaneous'] = 0

def category_counter(amenity, idx):
  for category, category_list in amenities_categorisation.items():
    for item in category_list:
      if item in amenities.lower():
        all_listings.loc[idx, category] += 1
        return True
  return False

unmapped_amenities = []
for i in range(all_listings.shape[0]):
  amenities_list = ast.literal_eval(all_listings.loc[i, 'amenities'])
  for amenities in amenities_list:
    mapped = category_counter(amenities, i)
    if not mapped:
      all_listings.loc[i, 'miscellaneous'] += 1 
      if amenities not in unmapped_amenities:
        unmapped_amenities.append(amenities)


# ### Availability

# There are multiple different measures of availability, which will be highly correlated with each other. Only one will be retained - for 30 days.

# In[36]:


all_listings.drop(['availability_60', 'availability_90', 'availability_365'], axis=1, inplace=True)


# ### Gender Split

# It would be interesting to see if the gender of the host affects the price

# In[37]:


import gender_guesser.detector as gender

d = gender.Detector()
all_listings['host_gender'] = all_listings['host_name'].apply(lambda x: d.get_gender(x))

gender_mapping = {'mostly_female': 'female', 'mostly_male': 'male', 'andy': 'unknown'}

all_listings = all_listings.replace({'host_gender': gender_mapping})


# ### Review Scores

# In[38]:


review_missing = all_listings[all_listings['review_scores_rating'].isna()]
reviews_columns = ['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'reviews_per_month']
review_missing[reviews_columns].isnull().sum()


# In[39]:


from pandas.api.types import CategoricalDtype

# high number of missing values, unfeasible to drop completely. imputation might not be valuable, hence best to encode as missing (logical as a consumer)

review_bins = []
review_bin_labels = []
columns_to_bin = ['review_scores_rating']

left, right = 0, 0.1
for i in range(50):
  review_bins += [left]
  label = str(left) + ' to ' + str(right)
  review_bin_labels += [label]
  left = round(left + 0.1, 1)
  right = round(right + 0.1, 1)
  if i == 49:
    review_bins += [5.1]

all_listings['review_scores_rating'] = pd.cut(all_listings['review_scores_rating'], bins=review_bins, labels=review_bin_labels, right=False)

all_listings['reviews_per_month'] = all_listings['reviews_per_month'].fillna(value=0)

cat_dtype = CategoricalDtype(categories=review_bin_labels+['NA'])
all_listings['review_scores_rating'] = all_listings['review_scores_rating'].astype(cat_dtype)
all_listings['review_scores_rating'] = all_listings['review_scores_rating'].fillna(value='NA')


# In[40]:


all_listings.drop(['review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value'], axis=1, inplace=True)


# ### Description Sentiment

# In[41]:


from textblob import TextBlob

def detect_sentiment(text):
  if pd.isnull(text):
    return -2
  blob = TextBlob(str(text))
  return blob.sentiment.polarity

all_listings['description_sentiment'] = all_listings.description.apply(detect_sentiment)
all_listings['neighborhood_overview_sentiment'] = all_listings.neighborhood_overview.apply(detect_sentiment)

bins = [-2, -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.01]
labels = ['NA', 'Very Negative', 'Negative', 'Slightly Negative', 'Neutral', 'Slightly Positive', 'Positive', 'Very Positive']

all_listings['description_sentiment'] = pd.cut(all_listings['description_sentiment'], bins, labels=labels, right=False)
all_listings['neighborhood_overview_sentiment'] = pd.cut(all_listings['neighborhood_overview_sentiment'], bins, labels=labels, right=False)


# ### Reviews Sentiment

# In[43]:


reviews = pd.read_csv("./Data/Reviews_w_sentiment.csv")
reviews.head()


# In[52]:


reviews.isna().sum()


# In[44]:


df = pd.merge(all_listings, reviews, left_on="id", right_on="listing_id", how="left")
df.shape


# In[49]:


df['sentiment_mean_score'].isna().sum()


# In[53]:


df['sentiment_median_score'].isna().sum()


# Since there's a couple of NA values, it means that certain listings don't have reviews or is not a part of the reviews dataset in airbnb. To counter this, we will fill NA values with a neutral sentiment.

# In[54]:


df['sentiment_mean_score'] = df['sentiment_mean_score'].fillna(0.0)
df['sentiment_median_score'] = df['sentiment_median_score'].fillna(0.0)


# ### Final check on columns

# In[56]:


df.info()


# Removing columns that are repeated or have been processed

# In[57]:


df.drop(['index','id','reviews_per_month', 'first_review', 'last_review', 'number_of_reviews_ltm', 'number_of_reviews_l30d', 'host_name','index','host_acceptance_rate'], axis=1, inplace=True)


# In[58]:


df.to_csv('cleaned_full_data.csv', index=False)
