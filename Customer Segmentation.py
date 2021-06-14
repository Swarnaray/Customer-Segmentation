#!/usr/bin/env python
# coding: utf-8

# ### 1. Importing the libraries and the data

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### 2. Importing the data from .csv file

# In[2]:


data = pd.read_csv('Customer.csv', encoding='unicode_escape')
data


# In[3]:


data.shape


# In[4]:


#droping all repeatative data from Country,CustomerID Columns
country_cust=data[['Country','CustomerID']].drop_duplicates()
#printing the existing data after droping and also sorting the data
country_cust.groupby(['Country'])['CustomerID'].aggregate('count').reset_index().sort_values('CustomerID', ascending=False)


# In[5]:


CC=country_cust.groupby(['Country'])['CustomerID'].aggregate('count').reset_index().sort_values('CustomerID', ascending=False)


# In[6]:


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(CC['Country'].head(5).values,CC['CustomerID'].head(5).values)
plt.show()


# In[7]:


#hence there is 90% of data is from UK so we are keeping only data of UK
data=data.query("Country=='United Kingdom'").reset_index(drop=True)
data.shape


# In[8]:


data.describe()


# ### 4. Checking the data for inconsistencies and further cleaning the data if needed.

# In[9]:


data.isnull()


# In[10]:


data.isnull().sum()


# In[11]:


#droping the null data
data=data[pd.notnull(data['CustomerID'])]


# In[12]:


data.isnull().sum()


# In[13]:


#validate if there any negative values of Quantity
data.Quantity.min()


# In[14]:


#removing all the negative values in the Quantity Column
data=data[(data['Quantity']>=0)]
data.Quantity.min()


# In[15]:


#validate if there any negative values of UnitPrice
data.UnitPrice.min()


# In[16]:


#calculating the total amount and storing the data into a newly added column
data['TotalAmount']=data['Quantity']*data['UnitPrice']
data


# In[17]:


data.InvoiceDate


# In[18]:


#Changing the data type of InvoiceDate from object to datetime
data['InvoiceDate']=pd.to_datetime(data['InvoiceDate'])
data.InvoiceDate


# In[19]:


#Searching for start and end date of this dataset
dataset=data.sort_values(['InvoiceDate'])
dataset


# In[20]:


import datetime as dt
#Hence we want to calculate recency we set the system date as the letest date from the dataset
Letest_Date=dt.datetime(2011,12,10)


# In[21]:


# Calculating the Recency,Frequency,Monetary
RFMScore=data.groupby('CustomerID').agg({'InvoiceDate': lambda x: (Letest_Date-x.max()).days, 'InvoiceNo': lambda x: len(x),
                                         'TotalAmount': lambda x: x.sum()})


# In[22]:


#Changing the 'InvoiceDate' datatype from datetime to integer
RFMScore['InvoiceDate']= RFMScore['InvoiceDate'].astype(int)


# In[23]:


#Renaming the Column names
RFMScore.rename(columns={'InvoiceDate':'Recency','InvoiceNo':'Frequency','TotalAmount':'Monetary'},inplace=True)


# In[24]:


RFMScore.head()


# In[25]:


RFMScore.Recency.describe()


# In[26]:


#Recency distribution plot
import seaborn as sns
x = RFMScore['Recency']

ax = sns.distplot(x)


# In[27]:


RFMScore.Frequency.describe()


# In[28]:


#Frequency distribution plot, taking observations which have frequency less than 1000
import seaborn as sns
x = RFMScore.query('Frequency < 1000')['Frequency']

ax = sns.distplot(x)


# In[29]:


RFMScore.Monetary.describe()


# In[30]:


#Monateray distribution plot, taking observations which have monetary value less than 10000
import seaborn as sns
x = RFMScore.query('Monetary < 10000')['Monetary']

ax = sns.distplot(x)


# In[31]:


#Creating Different quantile level to seggrigate the customer
quantiles=RFMScore.quantile(q=[0.2,0.4,0.6,0.8])
quantiles=quantiles.to_dict()


# In[32]:


quantiles


# In[33]:


#Defining Function to allote points based on the customers
def RScore(x,p,d):
    if x<=d[p][0.2]:
        return 4
    elif x<=d[p][0.4]:
        return 3
    elif x<=d[p][0.6]:
        return 2
    elif x<=d[p][0.8]:
        return 1
    else:
        return 0

#Defining Function to allote points based on the customers
def FScore(x,p,d):
    if x<=d[p][0.2]:
        return 1
    elif x<=d[p][0.4]:
        return 2
    elif x<=d[p][0.6]:
        return 3
    elif x<=d[p][0.8]:
        return 4
    else:
        return 5
    
#Defining Function to allote points based on the customers
def MScore(x,p,d):
    if x<=d[p][0.2]:
        return 1
    elif x<=d[p][0.4]:
        return 2
    elif x<=d[p][0.6]:
        return 3
    elif x<=d[p][0.8]:
        return 4
    else:
        return 5


# In[34]:


RFMScore['R']=RFMScore['Recency'].apply(RScore, args=('Recency',quantiles))
RFMScore['F']=RFMScore['Frequency'].apply(RScore, args=('Frequency',quantiles))
RFMScore['M']=RFMScore['Monetary'].apply(MScore, args=('Monetary',quantiles))


# In[35]:


RFMScore.head(7)


# In[36]:


RFMScore['RFMTotal']=RFMScore[['R','F','M']].sum(axis=1)
RFMScore.head(7)


# In[37]:


#Assign Loyalty Level to each customer
Loyalty_Level = ['BAD','AVERAGE', 'GOOD', 'VALUABLE', 'PREMIUME']
Score_cuts = pd.qcut(RFMScore.RFMTotal, q = 5, labels = Loyalty_Level)
RFMScore['RFM_Loyalty_Level'] = Score_cuts.values
RFMScore.reset_index().head()


# In[38]:


#Validate the data for RFMGroup = 111
RFMScore[RFMScore['RFM_Loyalty_Level']=='VALUABLE'].sort_values('Monetary', ascending=False).reset_index().head(10)


# In[39]:


import chart_studio as cs
import plotly.offline as po
import plotly.graph_objs as gobj

#Recency Vs Frequency
graph = RFMScore.query("Monetary < 50000 and Frequency < 2000")

plot_data = [
    gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'BAD'")['Recency'],
        y=graph.query("RFM_Loyalty_Level == 'BAD'")['Frequency'],
        mode='markers',
        name='BAD',
        marker= dict(size= 10,
            line= dict(width=1),
            color= 'black',
            opacity= 0.6
           )
    ),
        gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'AVERAGE'")['Recency'],
        y=graph.query("RFM_Loyalty_Level == 'AVERAGE'")['Frequency'],
        mode='markers',
        name='AVERAGE',
        marker= dict(size= 10,
            line= dict(width=1),
            color= 'red',
            opacity= 0.6
           )
    ),
        gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'GOOD'")['Recency'],
        y=graph.query("RFM_Loyalty_Level == 'GOOD'")['Frequency'],
        mode='markers',
        name='GOOD',
        marker= dict(size= 10,
            line= dict(width=1),
            color= 'yellow',
            opacity= 0.6
           )
    ),
    gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'VALUABLE'")['Recency'],
        y=graph.query("RFM_Loyalty_Level == 'VALUABLE'")['Frequency'],
        mode='markers',
        name='VALUABLE',
        marker= dict(size= 10,
            line= dict(width=1),
            color= 'green',
            opacity= 0.6
           )
    ),
    gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'PREMIUME'")['Recency'],
        y=graph.query("RFM_Loyalty_Level == 'PREMIUME'")['Frequency'],
        mode='markers',
        name='PREMIUME',
        marker= dict(size= 10,
            line= dict(width=1),
            color= 'blue',
            opacity= 0.6
           )
    ),
]

plot_layout = gobj.Layout(
        yaxis= {'title': "Frequency"},
        xaxis= {'title': "Recency"},
        title='Segments'
    )
fig = gobj.Figure(data=plot_data, layout=plot_layout)
po.iplot(fig)


# ## Recency Vs Monetary

# In[40]:


graph = RFMScore.query("Monetary < 50000 and Frequency < 2000")

plot_data = [
    gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'BAD'")['Recency'],
        y=graph.query("RFM_Loyalty_Level == 'BAD'")['Monetary'],
        mode='markers',
        name='BAD',
        marker= dict(size= 10,
            line= dict(width=1),
            color= 'black',
            opacity= 0.6
           )
    ),
        gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'AVERAGE'")['Recency'],
        y=graph.query("RFM_Loyalty_Level == 'AVERAGE'")['Monetary'],
        mode='markers',
        name='AVERAGE',
        marker= dict(size= 10,
            line= dict(width=1),
            color= 'red',
            opacity= 0.6
           )
    ),
        gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'GOOD'")['Recency'],
        y=graph.query("RFM_Loyalty_Level == 'GOOD'")['Monetary'],
        mode='markers',
        name='GOOD',
        marker= dict(size= 10,
            line= dict(width=1),
            color= 'yellow',
            opacity= 0.6
           )
    ),
    gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'VALUABLE'")['Recency'],
        y=graph.query("RFM_Loyalty_Level == 'VALUABLE'")['Monetary'],
        mode='markers',
        name='VALUABLE',
        marker= dict(size= 10,
            line= dict(width=1),
            color= 'green',
            opacity= 0.6
           )
    ),
    gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'PREMIUME'")['Recency'],
        y=graph.query("RFM_Loyalty_Level == 'PREMIUME'")['Monetary'],
        mode='markers',
        name='PREMIUME',
        marker= dict(size= 10,
            line= dict(width=1),
            color= 'blue',
            opacity= 0.6
           )
    ),
]

plot_layout = gobj.Layout(
        yaxis= {'title': "Monetary"},
        xaxis= {'title': "Recency"},
        title='Segments'
    )
fig = gobj.Figure(data=plot_data, layout=plot_layout)
po.iplot(fig)


# ## Frequency Vs Monetary

# In[41]:


graph = RFMScore.query("Monetary < 50000 and Frequency < 2000")

plot_data = [
    gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'BAD'")['Frequency'],
        y=graph.query("RFM_Loyalty_Level == 'BAD'")['Monetary'],
        mode='markers',
        name='BAD',
        marker= dict(size= 10,
            line= dict(width=1),
            color= 'black',
            opacity= 0.8
           )
    ),
        gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'AVERAGE'")['Frequency'],
        y=graph.query("RFM_Loyalty_Level == 'AVERAGE'")['Monetary'],
        mode='markers',
        name='AVERAGE',
        marker= dict(size= 10,
            line= dict(width=1),
            color= 'red',
            opacity= 0.6
           )
    ),
        gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'GOOD'")['Frequency'],
        y=graph.query("RFM_Loyalty_Level == 'GOOD'")['Monetary'],
        mode='markers',
        name='GOOD',
        marker= dict(size= 10,
            line= dict(width=1),
            color= 'yellow',
            opacity= 0.6
           )
    ),
    gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'VALUABLE'")['Frequency'],
        y=graph.query("RFM_Loyalty_Level == 'VALUABLE'")['Monetary'],
        mode='markers',
        name='VALUABLE',
        marker= dict(size= 10,
            line= dict(width=1),
            color= 'green',
            opacity= 0.6
           )
    ),
    gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'PREMIUME'")['Frequency'],
        y=graph.query("RFM_Loyalty_Level == 'PREMIUME'")['Monetary'],
        mode='markers',
        name='PREMIUME',
        marker= dict(size= 10,
            line= dict(width=1),
            color= 'blue',
            opacity= 0.6
           )
    ),
]

plot_layout = gobj.Layout(
        yaxis= {'title': "Monetary"},
        xaxis= {'title': "Frequency"},
        title='Segments'
    )
fig = gobj.Figure(data=plot_data, layout=plot_layout)
po.iplot(fig)


# In[42]:


#Handle negative and zero values so as to handle infinite numbers during log transformation
def handle_neg_n_zero(num):
    if num <= 0:
        return 1
    else:
        return num
#Apply handle_neg_n_zero function to Recency and Monetary columns 
RFMScore['Recency'] = [handle_neg_n_zero(x) for x in RFMScore.Recency]
RFMScore['Monetary'] = [handle_neg_n_zero(x) for x in RFMScore.Monetary]

#Perform Log transformation to bring data into normal or near normal distribution
Log_Tfd_Data = RFMScore[['Recency', 'Frequency', 'Monetary']].apply(np.log, axis = 1).round(3)


# In[43]:


#Data distribution after data normalization for Recency
Recency_Plot = Log_Tfd_Data['Recency']
ax = sns.distplot(Recency_Plot)


# In[44]:


#Data distribution after data normalization for Frequency
Frequency_Plot = Log_Tfd_Data.query('Frequency < 1000')['Frequency']
ax = sns.distplot(Frequency_Plot)


# In[45]:


#Data distribution after data normalization for Monetary
Monetary_Plot = Log_Tfd_Data.query('Monetary < 10000')['Monetary']
ax = sns.distplot(Monetary_Plot)


# In[46]:


from sklearn.preprocessing import StandardScaler

#Bring the data on same scale
scaleobj = StandardScaler()
Scaled_Data = scaleobj.fit_transform(Log_Tfd_Data)

#Transform it back to dataframe
Scaled_Data = pd.DataFrame(Scaled_Data, index = RFMScore.index, columns = Log_Tfd_Data.columns)


# In[47]:


from sklearn.cluster import KMeans

sum_of_sq_dist = {}
for k in range(1,15):
    km = KMeans(n_clusters= k, init= 'k-means++', max_iter= 1000)
    km = km.fit(Scaled_Data)
    sum_of_sq_dist[k] = km.inertia_
    
#Plot the graph for the sum of square distance values and Number of Clusters
sns.pointplot(x = list(sum_of_sq_dist.keys()), y = list(sum_of_sq_dist.values()))
plt.xlabel('Number of Clusters(k)')
plt.ylabel('Sum of Square Distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# In[48]:


#Perform K-Mean Clustering or build the K-Means clustering model
KMean_clust = KMeans(n_clusters= 3, init= 'k-means++', max_iter= 10000)
KMean_clust.fit(Scaled_Data)

#Find the clusters for the observation given in the dataset
RFMScore['Cluster'] = KMean_clust.labels_
RFMScore.head()


# In[49]:


from matplotlib import pyplot as plt
plt.figure(figsize=(7,7))

##Scatter Plot Frequency Vs Recency
Colors = ["red", "green", "blue"]
RFMScore['Color'] = RFMScore['Cluster'].map(lambda p: Colors[p])
ax = RFMScore.plot(    
    kind="scatter", 
    x="Recency", y="Frequency",
    figsize=(10,5),
    c = RFMScore['Color']
)


# In[50]:


from matplotlib import pyplot as plt
plt.figure(figsize=(7,7))

##Scatter Plot Recency Vs Monetary
Colors = ["red", "green", "blue"]
RFMScore['Color'] = RFMScore['Cluster'].map(lambda p: Colors[p])
ax = RFMScore.plot(    
    kind="scatter", 
    x="Recency", y="Monetary",
    figsize=(10,8),
    c = RFMScore['Color']
)


# In[51]:


from matplotlib import pyplot as plt
plt.figure(figsize=(7,7))

##Scatter Plot Monetary Vs Frequency
Colors = ["red", "green", "blue"]
RFMScore['Color'] = RFMScore['Cluster'].map(lambda p: Colors[p])
ax = RFMScore.plot(    
    kind="scatter", 
    x="Monetary", y="Frequency",
    figsize=(10,8),
    c = RFMScore['Color']
)

