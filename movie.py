#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np


# In[2]:


#read the data
movie_data = pd.read_csv("IMDb Movies India.csv" , encoding='latin1')
movie_data


# In[3]:


movie_data.head()


# In[4]:


movie_data.info()


# In[5]:


movie_data.describe()


# In[6]:


movie_data.isnull().sum()


# In[7]:


movie_data.dropna(subset["Rating"], inplace=True)


# In[8]:


movie_data.isnull().sum()


# In[9]:


movie_data.dropna(subset=['Actor1','Actor2','Actor3','Genre','Director'], inplace=True)


# In[10]:


movie_data.isnull().sum()


# In[11]:


movie_data.head()


# In[12]:


#convert votes columns
movie_data['Votes'] = movie_data['Votes'].str.replace(',','').astype(int)


# In[14]:


# convert year columns
movie_data['Year'] = movie_data['Year'].str.strip('()').astype(int)


# In[15]:


# convert duration columns
movie_data['Duration'] = movie_data['Duration'].str.strip('min')


# In[16]:


movie_data['Duration'].fillna (movie_data['Duration'].median(), inplace=True)


# In[17]:


movie_data.isnull().sum()


# In[18]:


movie_data.info()


# In[19]:


movie_data.head()


# In[20]:


#find top 10 movies based on rating
top_movie= movie_data.loc[movie_data['Rating'].sort_values(ascending=False) [:10].index]
top_movie


# In[21]:


sns.histplot(data=top_movie, x="Year", hue="Rating", multiple="stack")
plt.title('Distribution of Top 10 Movies', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Rating', fontsize=14)
plt.tight_layout()
plt.show()


# In[22]:


genre_counts= movie_data['Genre'].value_counts().reset_index()


# In[23]:


sns.histplot(data=top_movie, x="Year", hue="Rating", multiple="stack")
plt.title('Distribution of Top 10 Movies', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Rating', fontsize=14)
plt.tight_layout()
plt.show()


# In[24]:


genre_counts = movie_data['Genre'].value_counts().reset_index()
genre_counts.columns = ['Genre','Count']


# In[25]:


#select the top N genres
top_n_genres = genre_counts.head(5)
top_n_genres


# In[26]:


# Group the data by director and calculate the average rating.
director_avg_rating = movie_data.groupby('Director') ['Rating'].mean().reset_index()
director_avg_rating = director_avg_rating.sort_values (by='Rating', ascending=False)
top_directors =director_avg_rating.head (10)
top_directors


# In[27]:


plt.figure(figsize=(12, 6))
sns.barplot(data=top_directors, x='Rating', y='Director', palette='viridis')

plt.title('Top Directors by Average Rating', fontsize=16)
plt.xlabel('Average Rating', fontsize=14)
plt.ylabel('Director', fontsize=14)
plt.show()


# In[28]:


plt.figure(figsize=(12, 6))
sns.lineplot(data=movie_data, x='Rating', y='Votes')
plt.title('Votes vs. Rating', fontsize=16)
plt.xlabel('Rating', fontsize=14)
plt.ylabel('Votes', fontsize=14)
plt.show()


# In[29]:


actor_counts = movie_data['Actor 1'].value_counts().reset_index()
actor_counts.columns = ['Actor', " MovieCount"]
top_n_actors= actor_counts.head(10)
top_n_actors


# In[30]:


plt.figure(figsize=(12, 6)) 
sns.barplot(data=top_n_actors, x='MovieCount', y='Actor', orient='h')

# Set plot labels and title
plt.title('Top Actors by Number of Movies', fontsize=16)
plt.xlabel('Number of Movies', fontsize=14)
plt.ylabel('Actor', fontsize=14)

# Show the plot
plt.show


# In[31]:


yearly_movie_counts = movie_data['Year'].value_counts().reset_index()

yearly_movie_counts.columns = ['Year', 'MovieCount']
yearly_movie_counts= yearly_movie_counts.sort_values(by='Year')
yearly_movie_counts


# In[32]:


plt.figure(figsize=(12, 6)) 
sns.lineplot(data=yearly_movie_counts, x= 'Year', y='MovieCount')
plt.title('Number of Movies Released Every Year', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Number of Movies Released', fontsize=14)
plt.show()


# In[33]:


filtered_df = movie_data[(movie_data['Rating'] > 8) & (movie_data ['Votes'] > 10000)]
filtered_df.head(10)


# In[34]:


plt.figure(figsize=(15, 6))
ax=sns.barplot(data= filtered_df,x='Name', y='Votes', hue='Rating', dodge=False, width=0.5, palette='muted')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')
ax.legend(loc='upper right')
ax.set_xlabel('Movie Name')
ax.set_ylabel('Votes')
ax.set_title('Movies with rating greater than 8 and votes greater than 10000')
plt.show()


# In[35]:


movie_data['Duration'] = movie_data['Duration'].astype(int)
movie_data['Year'] = movie_data['Year'].astype(int)
                                
plt.figure(figsize=(12, 6))
sns.lineplot(data=movie_data,x='Year', y='Duration', errorbar=None)
plt.xlabel('Year')
plt.ylabel('Duration in minutes')
plt.title('Duration of movies by year')
plt.xticks(np.arange(1917,2023,5))
plt.show()


# In[36]:


movie_data['Genre'] = movie_data['Genre'].str.split(',')
#Create a new DataFrame with one row for each genre
genre_df = movie_data.explode('Genre')
genre_df


# In[37]:


plt.figure(figsize=(12, 6))
sns.countplot(data=genre_df, x='Genre', order=genre_df['Genre'].value_counts().index, palette='viridis')
plt.title('Number of Movies for Each Genre', fontsize=16)
plt.xlabel('Number of Movies', fontsize=14)
plt.ylabel('Genre', fontsize=14)
plt.xticks(rotation=90)
plt.show()


# In[38]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
trans_data =movie_data.drop(['Name'], axis=1)
#Transform Director columns
trans_data['Director'] = labelencoder.fit_transform(movie_data['Director'])
#Transform Actors Columns
trans_data['Actor 1'] = labelencoder.fit_transform(movie_data['Actor 1'])
trans_data['Actor 2'] = labelencoder.fit_transform(movie_data['Actor 2'])
trans_data['Actor 3'] = labelencoder.fit_transform(movie_data['Actor 3'])
trans_data['Genre'] = labelencoder.fit_transform(movie_data['Genre'].apply(lambda x: '.'.join(x)))
trans_data.head()


# In[48]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
sc_data=scaler.fit_transform(trans_data)
sc_df= pd.DataFrame (sc_data, columns=trans_data.columns)
sc_df.head()


# In[40]:


# correlation
corr_df = trans_data.corr(numeric_only=True)
corr_df['Rating'].sort_values (ascending=False)


# In[41]:


sns.heatmap(corr_df,annot=False, cmap="coolwarm")


# In[42]:


# Import modeling Libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[43]:


# put data except Rating data
x = trans_data.drop(['Rating'],axis=1)

# Put only Rating data y trans_data['Rating']
X_train, X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=42)
model =LinearRegression()
model.fit(X_train,y_train)


# In[44]:


X_test = np.array(X_test)


# In[45]:


y_pred model.predict(X_test) y_pred


# In[46]:


print('R2 score: ',r2_score(y_test,y_pred))
print('Mean squared error: ', mean_squared_error(y_test,y_pred))
print('Mean absolute error: ', mean_absolute_error(y_test,y_pred))


# In[47]:


print(y_test)


# In[ ]:





# In[ ]:




