# %% [markdown]
# # Used car analysis 
# ##### _Mario Krivosic_
# 

# %%
#import neccesary packages for extraction and graphing
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import seaborn as sbn
import scipy as sp
import requests
%matplotlib inline

# %%
#Get data
file_path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'
data = requests.get(file_path)



# %%
#convert it to a file
with open ('download', 'wb') as file:
    file.write(data.content)

# %%
#read and trasnform data to pandas 
df = pd.read_csv('download', header = 0)

# %%
#check the first rows of data to ensure they are fine
df.head()

# %%
#ensure all data types are corrext 
print(df.dtypes)

# %%
#check correleation between variables and their types
df[['bore', 'stroke', 'compression-ratio', 'horsepower', 'engine-size', 'price']].corr()

# %%
#plot the idnetified high correlation between price and engine-size
sbn.regplot(x="engine-size", y = "price", data=df)
plt.ylim(1,)  #plot line which showcases trend
plt.show()

# %%
#plot another statistical relationship which was shown to have some correlation
sbn.regplot(x='stroke', y = 'horsepower', data = df)
plt.ylim(1,)
plt.show()
# data  shows a weak correleation 

# %% [markdown]
# # Categorical Value analysis
# 

# %%
# relationship between categorical value and price
sbn.boxplot(x='body-style', y = 'price', data = df)
plt.show()

# %%
#asses similiar relationship with wheels
sbn.boxplot(x = 'drive-wheels', y = 'price', data = df)
plt.show()

# %% [markdown]
# # Descriptive tests

# %% [markdown]
# Look into the data more thoroughly to understand it better

# %%
#get overview of data just integers
df.describe()

# %%
#overview of dara with objects 
df.describe(include = ['object'])

# %%
#check and convert drive wheeks
df['drive-wheels'].value_counts()
drive_wheels_count = df['drive-wheels'].value_counts().to_frame()

# %%
#add this to column and rename 
drive_wheels_count.reset_index(inplace = True)
drive_wheels_count = drive_wheels_count.rename(columns={'drive-wheels':'Value_counts'})
drive_wheels_count

# %%
#rename the index column
drive_wheels_count.index.name = 'drive-wheels'
drive_wheels_count

# %%
#create seperate database for engine 
engine_loc = df['engine-location'].value_counts().to_frame()
engine_loc.rename(columns={'engine-location':'value_count'})
engine_loc.index.name = 'engine-location'
engine_loc.head()

# %% [markdown]
# ## initial findings  
# This sugest engine_location is a bad predictor of price due to possible skewness caused by low number of rear cars and drive wheels while hvent adequete splits between fwd, rwd however 4wd is under represented

# %%
 #group wheels to find relationship with price 
df_wheel_group = df[['drive-wheels', 'body-style', 'price']]

# %%
#add it to a group by and look at mean exchanging price as the name 
df_grouped = df_wheel_group.groupby(['drive-wheels'], as_index= False).agg({'price':'mean'})
df_grouped

# %%
# lets expand the analysis by adding body-style 
df_grouped = df_wheel_group.groupby(['drive-wheels', 'body-style'], as_index= False).agg({'price':'mean'})
df_grouped

# %%
#convert this to a pivot table 
group_pivot = df_grouped.pivot(index = 'drive-wheels', columns = 'body-style')
group_pivot

# %%
# fill the null values with 0 to allow analysis
group_pivot = group_pivot.fillna(0)
group_pivot

# %%
#visualise with a heat map
fig, ax = plt.subplots()
im = ax.pcolor(group_pivot, cmap = 'RdBu')

#Labeling heatmap
row_labels = group_pivot.columns.levels[1]
col_labels = group_pivot.index
#move labels to the middle 
ax.set_xticks(np.arange(group_pivot.shape[1]) + 0.5, minor = False)
ax.set_yticks(np.arange(group_pivot.shape[0]) + 0.5, minor = False)

#insert labels
ax.set_xticklabels(row_labels)
ax.set_yticklabels(col_labels)
ax.set_title('Heatmap of Price')
plt.xticks(rotation = 90)

fig.colorbar(im)
plt.show()





# %% [markdown]
# # Utilising p-value

# %%
#import package for scientific 
from scipy import stats 

# %%
# getting pearson coeffecient 
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print(f"The pearson coeffecint: {pearson_coef}", f"the p-value: {p_value}")

# %% [markdown]
# THe p-value is less then 1e-4 the correleation is significant 
# The pearson coeffecient suggest a modertly strong _positive_ linear relationship

# %%
#getting some more pearsons and p-value for other variable
pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])
print(f"The pearson coeffecint: {pearson_coef}", f"the p-value: {p_value}")

# %% [markdown]
# p-Value suggest strong coeraltion 
# Pearson shows stronger negative linear relationship

# %% [markdown]
# 


