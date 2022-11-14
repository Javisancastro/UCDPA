#!/usr/bin/env python
# coding: utf-8

# In[1]:


#1 DATA- Data publicly available at - https://www.metacritic.com/browse/games/score/metascore/all


# In[2]:


#2 IMPORTING DATA PACKAGES 


# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


#Importing Games CSV File - Read the data from the dataset
games = pd.read_csv("games.csv")
print (games)


# In[5]:


# Gathering info about the dataset - shape
games.shape


# In[6]:


# Gathering info about the dataset - info
games.info()


# In[7]:


#3 PREPARING THE DATA FOR ANALYSIS - DATA CLEANING


# In[8]:


# Formula for unifying data - Round meta_score  

games["meta_score"] = games["meta_score"]/10

# Checking if it works

print(games)


# In[9]:


#Identifying blank spaces in columns and values
print(games.isna().sum())


# In[10]:


#There are missing values in the "summary", replacing blanks with None
games = games.fillna("None")


# In[11]:


#Checking if there are still missing values
print(games.isna().sum())


# In[12]:


# remove whitespaces from platform
games["platform"].str.replace(' ', '')


# In[13]:


#Checking for duplicates in the dataset
duplicate_games =games.duplicated().any()
print(duplicate_games)


# In[14]:


#Adding a new column that shows the difference between the meta_score and user_review & the grouped between the meta_score and user_review 
games["discrepancy"] = games ["meta_score"]-games["user_review"]
games["grouped"] = (games ["meta_score"] + games["user_review"])/2


# In[15]:


# Displaying  a preview of the data.
games.head(17435)


# In[16]:


#SORTING THE DATA 


# In[34]:


# Sorting by year
games_year = games.sort_values("year", ascending=False)
games_year.head(17435)


# In[69]:


# Sorting by user_review
games_user_review = games.sort_values("user_review", ascending=False)
games_user_review.head(10)


# In[70]:


# Sorting by meta_score
games_meta_score = games.sort_values("meta_score", ascending=False)
games_meta_score.head(10)


# In[73]:


# Sorting by Grouped /  Top 10
games_grouped = games.sort_values("grouped", ascending=False)
games_grouped.head(10)


# In[19]:


# Setting an index and sorting based on that index before slicing
games_srt = games.set_index("year", drop=False).sort_index()

#Slicing by a set period of time - Getting data for games released between 1995 and 2009, and between 2010 and 2021
games_old = games_srt.loc["1995":"2009"]
games_recent = games_srt.loc["2010":"2021"]

print(games_old)
print(games_recent)


# In[20]:


# Counting the number of games for each date range
print(str(games_old["year"].count()) + " games released between 1995 and 2009, and " + str(games_recent["year"].count()) + " games released between 2010 and 2021.")


# In[77]:


# Sorting by Grouped "old games" / Top 5
games_grouped_old = games_old.sort_values("grouped", ascending=False)
games_grouped_old.head(5)


# In[76]:


# Sorting by Grouped "recent games" / Top 5
games_grouped_recent = games_recent.sort_values("grouped", ascending=False)
games_grouped_recent.head(5)


# In[21]:


#Merging data frames (reusing previous slicing to go back to original data)

dataframes = [games_old, games_recent]
result = pd.concat(dataframes)
print (games_srt)
print (result)


# In[22]:


#ANALIZING THE DATA


# In[23]:


#General info about the data

print(games.describe())


# In[59]:


# - What is the total number of video games per platform?
total_games_platform = games[["platform","name"]].groupby("platform").count()
total_games_platform = total_games_platform.sort_values ("name",ascending=False )
print(total_games_platform)


# In[ ]:





# In[78]:


# - What is the total number of video games per year?
total_games_year = games[["name","year"]].groupby("year").count()
total_games_year = total_games_year.sort_values ("name",ascending=False )
print(total_games_year)


# In[ ]:





# In[26]:


#Histogram about distributions of meta_score reviews

median_length = games["meta_score"].median()
print(median_length)
color_hist = "#A23B72"
color_line = "#2E86AB"
plt.hist(games["meta_score"], bins=20, color=color_hist, edgecolor="black")
plt.axvline(median_length, color=color_line, label="Median Length")
plt.legend()
plt.title("Distribution of meta_score reviews")
plt.xlabel("Game score")
plt.ylabel("Number of games")
plt.grid(axis="y", color="gray", linestyle="-", linewidth=0.7)
plt.tight_layout()
plt.savefig("fig1.png")
plt.show()


# In[27]:


# It looks like most games tend do be reviewed between 7 and 8, which means that they are good to play


# In[45]:


#Histogram about distributions of user_review reviews

median_length = games["user_review"].median()
print(median_length)
color_hist = "#A23B72"
color_line = "#2E85AB"
plt.hist(games["user_review"], bins=20, color=color_hist, edgecolor="black")
plt.axvline(median_length, color=color_line, label="Median Length")
plt.legend()
plt.title("Distribution of user_review reviews")
plt.xlabel("Game score")
plt.ylabel("Number of games")
plt.grid(axis="y", color="gray", linestyle="-", linewidth=0.7)
plt.tight_layout()
plt.savefig("fig2.png")
plt.show()


# In[43]:


# It looks like the user_review reviews follow the same trend as Meta_score, between 7 and 8


# In[30]:


# Is there a correlation between critic reviews (meta_score) & user reviews?

correlation =games["meta_score"].corr(games["user_review"])
print(correlation)


# In[31]:


# Creating a scatter plot graph to compare the variables.
plt.figure(figsize=(15, 10))
sns.set_style('whitegrid')
fig = sns.regplot(x="user_review", y="meta_score", data=games, scatter_kws={'color': '#550527', 'alpha': 0.4},
                  line_kws={'color': '#688E26', 'alpha': 0.4})
plt.xlabel("User Score", fontsize=13)
plt.ylabel('Meta Score', fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title('Correlation between User & Meta score', fontsize=20)
plt.savefig('fig3.png')
plt.show()


# In[32]:


# Are games getting better according to users?  - Linear chart 



plt.figure(figsize=(10, 10))
sns.set_style("dark")
fig2 = sns.lineplot(x='year', y= ("user_review"), data=games)
fig2.set(xlabel="Year", ylabel="User_review")
fig2.set_title('Are games getting better according to users', size=18)
plt.savefig('fig4.png')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:


#Merge data frames using Pandas .pd.merge / (this code is not related to the main dataset, but I am executing in to complete one of the tasks in this assessment)


# In[83]:


df1 = pd.DataFrame({
    "Country": [ 'Russia', 'Ireland', 'Spain'],
    "Capital": ['Moscow' , 'Dublin', 'Madrid']
})


# In[87]:


df2 = pd.DataFrame({
    "Country": ['France', 'Spain', 'Russia'],
    "Currency": ['Euro', 'Euro', 'Ruble']
    })


# In[88]:


df3 = pd.merge(df1, df2, on='Country')
print(df3)


# In[ ]:





# In[ ]:




