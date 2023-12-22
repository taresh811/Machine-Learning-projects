#!/usr/bin/env python
# coding: utf-8

# The objective is to analyze the flight booking dataset obtained from a platform which is used to book flight
# tickets. A thorough study of the data will aid in the discovery of valuable insights that will be of enormous
# value to passengers. Applying EDA, statistical methods and Machine learning algorithms in order to get
# meaningful information from it.

# In[25]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# In[26]:


# Looking at the data and removing unnecessary column from the dataframe  
df = pd.read_csv("Flight_Booking.csv")
df = df.drop(columns=["Unnamed: 0"])
df.head()


# In[27]:


# Checking the shape of the dataframe and datatypes of all columns along with calculating the statistical data
df.shape


# In[28]:


df.info()


# In[50]:


df.describe(include='all')


# In[30]:


# Checking out the missing values in the dataframe
df.isnull().sum()


# In[31]:


plt.figure(figsize = (15,5))
sns.lineplot(x=df['airline'],y=df['price'])
plt.title('Airline vs Price',fontsize =15)
plt.xlabel('Airline',fontsize=15)
plt.ylabel('Price',fontsize=15)
plt.show()


# There is a variation in price with different airlines
# 

# In[32]:


plt.figure(figsize = (15,5))
sns.lineplot(data=df,x='days_left',y='price',color='blue')
plt.title('Days left for departure vs Ticket price',fontsize =15)
plt.xlabel('Days left for departure',fontsize=15)
plt.ylabel('Price',fontsize=15)
plt.show()


# The price of the ticket increases as the days left for departure decreases 

# In[33]:


plt.figure(figsize=(10,5))
sns.barplot(x='airline',y='price',data=df)


# Price range of all the flights 

# In[34]:


plt.figure(figsize=(10,5))
sns.barplot(x='class',y='price',data=df,hue='airline')


# Range of price of all the flights of Economy and Business class

# In[35]:


fig,ax=plt.subplots(1,2,figsize=(20,6))
sns.lineplot(data=df,x='days_left',y='price',hue='source_city',ax=ax[0])
sns.lineplot(data=df,x='days_left',y='price',hue='destination_city',ax=ax[1])


# Range of price of flights with source and destination city according to the days left 

# In[36]:


# Visualization of categorical features with countplot
plt.figure(figsize=(15,23))

plt.subplot(4,2,1)
sns.countplot(x=df['airline'],data=df)
plt.title('Frequency of Airline')

plt.subplot(4,2,2)
sns.countplot(x=df['source_city'],data=df)
plt.title('Frequency of Source City')

plt.subplot(4,2,3)
sns.countplot(x=df['departure_time'],data=df)
plt.title('Frequency of Departure time')

plt.subplot(4,2,4)
sns.countplot(x=df['stops'],data=df)
plt.title('Frequency of Stops')

plt.subplot(4,2,5)
sns.countplot(x=df['arrival_time'],data=df)
plt.title('Frequency of Arrival time')

plt.subplot(4,2,6)
sns.countplot(x=df['destination_city'],data=df)
plt.title('Frequency of Destination City')

plt.subplot(4,2,7)
sns.countplot(x=df['class'],data=df)
plt.title('Class Frequency')

plt.show()


# In[37]:


# Performing One Hot Encoding for categorical features in the dataframe 

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['airline'] = le.fit_transform(df['airline'])
df['source_city'] = le.fit_transform(df['source_city'])
df['departure_time'] = le.fit_transform(df['departure_time'])
df['stops'] = le.fit_transform(df['stops'])
df['arrival_time'] = le.fit_transform(df['arrival_time'])
df['destination_city'] = le.fit_transform(df['destination_city'])
df['class'] = le.fit_transform(df['class'])
df.info()


# In[39]:


# plotting the correlation graph to see the correlation between features and dependent variable 

plt.figure(figsize=(10,5))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
plt.show()


# In[45]:


# selecting the features using VIF . VIF should be less than 5 thus dropping the features.

from statsmodels.stats.outliers_influence import variance_inflation_factor
col_list = []
for col in df.columns:
    if ((df[col].dtype != 'object') & (col != 'price') ):
        col_list.append(col)
        
X = df[col_list]
vif_data = pd.DataFrame()
vif_data['features'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data)


# In[46]:


df=df.drop(columns=['stops'])

# selecting the features using VIF . VIF should be less than 5 thus dropping the features.

from statsmodels.stats.outliers_influence import variance_inflation_factor
col_list = []
for col in df.columns:
    if ((df[col].dtype != 'object') & (col != 'price') ):
        col_list.append(col)
        
X = df[col_list]
vif_data = pd.DataFrame()
vif_data['features'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data)


# In[62]:


# Applying standarization implementing Linear Regression Model to predict the price of a flight

X = df.drop(columns=['price','flight'])
y = df['price']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[61]:


#Implementing Linear Regression Model to predict the price of a flight

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
difference = pd.DataFrame(np.c_[y_test,y_pred],columns=['Actual_Value','Predicted_Value'])
difference


# In[53]:


#Calculating r2 score

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[55]:


#Calculating MAE

from sklearn import metrics
metrics.mean_absolute_error(y_test,y_pred)


# In[56]:


#Calculating MAPE

from sklearn.metrics import mean_absolute_percentage_error
mean_absolute_percentage_error(y_test,y_pred)


# In[58]:


#Calculating MSE

metrics.mean_squared_error(y_test,y_pred)


# In[59]:


#Calculating RMSE

np.sqrt(metrics.mean_squared_error(y_test,y_pred))


# Root Mean square error(RMSE)
# of the Linear regression model is 7259.93 and Mean absolute percentage
# error(MAPE) is 34 percent. Lower the RMSE and MAPE better the model

# In[60]:


# plotting the graph of actual & predicted price of flight

sns.distplot(y_test,label='Actual')
sns.distplot(y_pred,label='predicted')
plt.legend()


# In[63]:


# Decision Tree Regressor

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)


# In[64]:


#Calculating r2 score

r2_score(y_test,y_pred)


# In[65]:


#Calculating MAE

metrics.mean_absolute_error(y_test,y_pred)


# In[69]:


#Calculating  MAPE

mean_absolute_percentage_error(y_test,y_pred)*100


# In[67]:


#Calculating  MSE

metrics.mean_squared_error(y_test,y_pred)


# In[68]:


#Calculating RMSE

np.sqrt(metrics.mean_squared_error(y_test,y_pred))


# Mean absolute percentage error is 7.7 percent and RMSE
# is 3615 which is less than the linear regression model

# In[70]:


# plotting the graph of actual & predicted price of flight

sns.distplot(y_test,label='Actual')
sns.distplot(y_pred,label='predicted')
plt.legend()


# In[71]:


# Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(x_train,y_train)
y_pred = rfr.predict(x_test)


# In[72]:


#Calculating r2 score

r2_score(y_test,y_pred)


# In[73]:


#Calculating MAE

metrics.mean_absolute_error(y_test,y_pred)


# In[79]:


#Calculating  MAPE

mean_absolute_percentage_error(y_test,y_pred)*100


# In[75]:


#Calculating  MSE

metrics.mean_squared_error(y_test,y_pred)


# In[76]:


#Calculating RMSE

np.sqrt(metrics.mean_squared_error(y_test,y_pred))


# Mean absolute percentage error is 7.3 percent and RMSE
# is 2824 which is less than the linear regression and decision
# tree model

# In[80]:


# plotting the graph of actual & predicted price of flight

sns.distplot(y_test,label='Actual')
sns.distplot(y_pred,label='predicted')
plt.legend()


# In[ ]:




