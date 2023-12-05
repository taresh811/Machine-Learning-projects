#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Netflix prize dataset
#importing the necessary libraries for importing the dataset
#Around100M+ ratings 4499 movies 480,000 users
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


# Reading dataset file
netflix_dataset = pd.read_csv('combined_data_1.txt',header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])
netflix_dataset.head()


# In[4]:


netflix_dataset


# In[5]:


netflix_dataset.dtypes


# In[6]:



# netflix_dataset['Rating']=netflix_dataset['Rating'].astype(float)


# In[69]:


netflix_dataset.dtypes


# In[7]:



netflix_dataset.shape


# In[8]:


#get the customer count with NaN values
movie_count=netflix_dataset.isnull().sum()
movie_count


# In[9]:


#to claculate how many customers we are having in the dataset
customer_count=netflix_dataset['Cust_Id'].nunique()


# In[10]:


customer_count


# In[11]:


#without NaN values
customer_count=netflix_dataset['Cust_Id'].nunique()-movie_count
customer_count


# In[12]:


#get the total number of ratings given by the customers
rating_count=netflix_dataset['Cust_Id'].count()-movie_count
rating_count


# In[13]:


#To find out how many people have rated the movies as 1, 2, 3,4,5 stars ratings to the movies
stars=netflix_dataset.groupby('Rating')['Rating'].agg(['count'])


# In[14]:


ax=stars.plot(kind='barh', legend=False, figsize=(15,10))
plt.title(f'Total pool: {movie_count} Movies, {customer_count} Customers, {rating_count} ratings given', fontsize=20)
plt.grid(True)


# In[15]:


#add another column that will have movie id
#first of all we will be calculating how many null values we are having in the ratings column
df_nan=pd.DataFrame(pd.isnull(netflix_dataset.Rating))


# In[16]:


df_nan


# In[17]:


df_nan=df_nan[df_nan['Rating']==True]
df_nan


# In[18]:


df_nan.shape


# In[19]:


df_nan.head()


# In[20]:


df_nan.tail()


# In[21]:


#now we will reset the index as the column
df_nan=df_nan.reset_index()


# In[24]:


df_nan


# In[25]:


#To create a numpy array containing movie ids according the 'ratings' dataset

movie_np = []
movie_id = 1

for i,j in zip(df_nan['index'][1:],df_nan['index'][:-1]):
    # numpy approach
    temp = np.full((1,i-j-1), movie_id)
    movie_np = np.append(movie_np, temp)
    movie_id += 1


# In[ ]:


df_nan.iloc[-1, 0]


# In[ ]:


len(netflix_dataset)


# In[ ]:


# Account for last record and corresponding length
# numpy approach
last_record = np.full((1,len(netflix_dataset) - df_nan.iloc[-1, 0] - 1),movie_id)
movie_np = np.append(movie_np, last_record)

print(f'Movie numpy: {movie_np}')
print(f'Length: {len(movie_np)}')


# In[ ]:


len(netflix_dataset)


# In[29]:


netflix_dataset


# In[30]:


df_nan['index'][1:]


# In[31]:


df_nan['index'][:-1]


# In[32]:


len(netflix_dataset)


# In[33]:


#working
x=zip(df_nan['index'][1:], df_nan['index'][:-1])


# In[34]:


tuple(x)


# In[136]:


temp=np.full((1,145), 2) #(shape, fill_value)


# In[137]:


print(temp)


# In[92]:


netflix_dataset=netflix_dataset[pd.notnull(netflix_dataset['Rating'])]
netflix_dataset['Movie_Id']=movie_np.astype(int)
netflix_dataset['Cust_Id']=netflix_dataset['Cust_Id'].astype(int)
print("Now the dataset will look like: ")
netflix_dataset.head()


# In[93]:


netflix_dataset.tail()


# In[94]:


#now we will remove all the users that have rated less movies and
#also all those movies that has been rated less in numbers
f=['count','mean']


# In[95]:


dataset_movie_summary=netflix_dataset.groupby('Movie_Id').agg(f)


# In[96]:


dataset_movie_summary


# In[97]:


dataset_movie_summary=netflix_dataset.groupby('Movie_Id')['Rating'].agg(f)


# In[98]:


dataset_movie_summary


# In[99]:


#now we will store all the movie_id indexes in a variable dataset_movie_summary.index and convert the datatype to int
# dataset_movie_summary.index=dataset_movie_summary.index.map(int)


# In[100]:


dataset_movie_summary["count"].quantile(0.7)


# In[101]:


#now we will create a benchmark
movie_benchmark=round(dataset_movie_summary['count'].quantile(0.7),0)
movie_benchmark


# In[102]:


dataset_movie_summary['count']


# In[103]:


drop_movie_list=dataset_movie_summary[dataset_movie_summary['count']<movie_benchmark].index
drop_movie_list


# In[104]:


#now we will remove all the users that are in-active
dataset_cust_summary=netflix_dataset.groupby('Cust_Id')['Rating'].agg(f)
dataset_cust_summary


# In[105]:


# dataset_cust_summary.index=dataset_cust_summary.index.map(int)


# In[106]:


cust_benchmark=round(dataset_cust_summary['count'].quantile(0.7),0)
cust_benchmark


# In[107]:


drop_cust_list=dataset_cust_summary[dataset_cust_summary['count']<cust_benchmark].index
drop_cust_list


# In[108]:


#we will remove all the customers and movies that are below the benchmark
print('The original dataframe has: ', netflix_dataset.shape, 'shape')


# In[109]:


netflix_dataset=netflix_dataset[~netflix_dataset['Movie_Id'].isin(drop_movie_list)]
netflix_dataset=netflix_dataset[~netflix_dataset['Cust_Id'].isin(drop_cust_list)]
print('After the triming, the shape is: {}'.format(netflix_dataset.shape))


# In[ ]:





# In[110]:


netflix_dataset.head()


# In[111]:


#now we will prepare the dataset for SVD and it takes the matrix as the input
# so for input, we will convert the dataset into sparse matrix
#4499 movies
# df_p = pd.pivot_table(netflix_dataset, values='Rating', index='Cust_Id', columns='Movie_Id')
# print(df_p.shape)


# In[113]:


import pandas as pd


# In[114]:


df_title = pd.read_csv("movie_titles.csv",  encoding='ISO-8859-1', header=None, usecols=[0,1,2], names=['Movie_Id','Year','Name' ])

df_title.set_index('Movie_Id', inplace=True)


# In[115]:


df_title.head(10)


# In[116]:


get_ipython().system('pip install scikit-surprise')


# In[117]:


#model building

import math
import seaborn as sns
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate


# In[118]:


#!pip install scikit-surprise


# In[119]:


#help us to read the dataset for svd algo
reader=Reader()


# In[120]:


#we only work with top 100K rows for quick runtime
data=Dataset.load_from_df(netflix_dataset[['Cust_Id','Movie_Id','Rating']][:100000], reader)


# In[142]:


svd=SVD()
cross_validate(svd, data, measures=['RMSE','MAE'],verbose=True)
#for 1st fold- 1,2,3,4,5


# In[122]:


netflix_dataset.head()


# In[123]:


#so first we take user 712664 and we try to recommend some movies based on the past data
#He rated so many movies with 5 *
dataset_712664=netflix_dataset[(netflix_dataset['Cust_Id'] ==712664)& (netflix_dataset['Rating']==5)]
# dataset_712664=dataset_712664.set_index('Movie_Id')
# dataset_712664=dataset_712664.join(df_title)['Name']
dataset_712664


# In[124]:


df_title


# In[125]:


#now we will build the recommendation algorithm
#first we will make a shallow copy of the movie_titles.csv file so that we can change
#the values in the copied dataset, not in the actual dataset

user_712664=df_title.copy()
user_712664


# In[126]:


user_712664=user_712664.reset_index()
user_712664


# In[127]:


user_712664=user_712664[~user_712664['Movie_Id'].isin(drop_movie_list)]
user_712664


# In[128]:


user_712664['Estimate_Score']=user_712664['Movie_Id'].apply(lambda x: svd.predict(712664, x).est)
user_712664=user_712664.drop('Movie_Id', axis=1)


# In[129]:


user_712664=user_712664.sort_values('Estimate_Score')
print(user_712664)


# In[130]:


# user_712664.head(10)


# In[131]:


user_712664=user_712664.sort_values('Estimate_Score', ascending=False)
print(user_712664.head(10))

