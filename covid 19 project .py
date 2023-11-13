#!/usr/bin/env python
# coding: utf-8

# # Project – Analyzing the trends of COVID-19 with Python
#  
# Problem Statement: Given data about COVID 19 patients, write code to visualize the impact and analyze the trend of rate of infection and recovery as well as make predictions about the number of cases expected a week in future based on the current trends
# 
# Guidelines: 
# • Use pandas to accumulate data from multiple data files
# • Use plotly (visualization library) to create interactive visualizations 
# • Use Facebook prophet library to make time series models 
# • Visualize the prediction by combining these technologies

# In[1]:


#importing all the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px #chloropleth

import warnings
warnings.filterwarnings("ignore")


# In[2]:


#load the data
data = pd.read_csv("covid_19_clean_complete.csv", parse_dates=['Date'])


# In[3]:


data.head()


# In[4]:


data.rename(columns={'Date':'date','Province/State':'state','Country/Region':'country','Lat':'lat','Long':'long',
                     'Confirmed':'confirmed','Deaths':'deaths','Recovered':'recovered','Active':'active'},inplace=True)


# In[24]:


data.head()


# In[6]:


data['date'].value_counts()


# In[7]:


data['date'].max()


# In[8]:


#To find all the cases on last value in date
top = data[data['date'] == data['date'].max()]
top


# In[9]:


w = top.groupby('country')['confirmed','active','deaths'].sum()
w


# In[10]:


w = w.reset_index()
w


# In[11]:


#choropleth - A choropleth map is a thematic map that is used to represent statistical data
#using the color mapping symbology technique.

fig=px.choropleth(w,locations='country',locationmode='country names',color='active',hover_name='country',
                 range_color=[1,1500],color_continuous_scale="Peach",title='Active cases Countries')
fig.show()


# In[25]:


#plot for confirmed cases
plt.figure(figsize=(15,10))

t_cases = data.groupby('date')['confirmed'].sum().reset_index()
t_cases['date'] = pd.to_datetime(t_cases['date'])

a=sns.pointplot(x=t_cases['date'].dt.date.head(50),y=t_cases['confirmed'].head(50),color='r')
a.set(xlabel='Dates',ylabel='Cases total')

plt.xticks(rotation=90,fontsize=10)
plt.yticks(fontsize=15)
plt.xlabel('Dates',fontsize=10)
plt.ylabel('Cases total',fontsize=30)


# In[13]:


top


# In[14]:


#grouping the top dataset by country column and sorting te result in descending and findind the to 20
t_actives=top.groupby(by='country')['active'].sum().sort_values(ascending=False).head(20).reset_index()
t_actives


# In[15]:


plt.figure(figsize=(15,10))
plt.title('Top 20 countries having most active cases',fontsize=30)
#barplot

a = sns.barplot(x=t_actives.active, y=t_actives.country)

plt.xticks(fontsize=20) #x label
plt.yticks(fontsize=20)
plt.xlabel('Cases total',fontsize=20)
plt.ylabel('Countryl',fontsize=20)


# In[16]:


Brazil = data[data.country=="Brazil"]
Brazil = Brazil.groupby('date')['recovered','deaths','confirmed','active'].sum().reset_index()
Brazil


# In[17]:


pip install prophet


# In[18]:


from prophet import Prophet


# In[19]:


data.head()


# In[20]:


data.groupby("date").sum().head()


# In[21]:


confirmed = data.groupby("date")['confirmed'].sum().reset_index()
confirmed.head()


# In[22]:


deaths = data.groupby("date")['deaths'].sum().reset_index()
recovered = data.groupby("date")['recovered'].sum().reset_index()


# In[23]:


confirmed.head()


# In[26]:


confirmed.columns = ['ds', 'y']
confirmed['ds']= pd.to_datetime(confirmed['ds'])
confirmed.head()


# In[27]:


m = Prophet()
m.fit(confirmed) #training the model


# In[28]:


confirmed.tail()


# In[29]:


future = m.make_future_dataframe(periods=7, freq='D')
future.tail(20)


# In[30]:


forecast = m.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']]


# In[31]:


confirmed_cases_forecast = m.plot(forecast)


# In[32]:


forecast['yhat'] = forecast['yhat'].astype(int)


# In[33]:


forecast[['ds','yhat']].tail(12)


# Forecast on Death case

# In[34]:


deaths.head()


# In[35]:


deaths.columns = ['ds', 'y']
deaths['ds']= pd.to_datetime(deaths['ds'])


# In[36]:


m = Prophet()
m.fit(deaths) #training
future = m.make_future_dataframe(periods = 21)
future.tail()


# In[37]:


forecast = m.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()


# In[38]:


deaths_forecast_plot = m.plot(forecast)


# In[39]:


forecast['yhat'] = forecast['yhat'].astype(int)


# In[40]:


forecast[['ds','yhat']].tail(25)


# Forecast on Recovered Cases

# In[41]:


recovered.columns = ['ds', 'y']
recovered['ds']= pd.to_datetime(recovered['ds'])


# In[42]:


recovered


# In[43]:


m = Prophet()
m.fit(recovered)
future=m.make_future_dataframe(periods=21)
future.tail()


# In[44]:


forecast = m.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']]


# In[45]:


recovered_forecast_plot = m.plot(forecast)


# In[46]:


forecast['yhat'] = forecast['yhat'].astype(int)


# In[47]:


forecast[['ds','yhat']].tail(25)


# In[ ]:




