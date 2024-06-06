# Capstone_recommender_sys
'''
# python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

tourism_with_id = pd.read_excel('tourism_with_id.xlsx')
tourism_rating = pd.read_csv('tourism_rating.csv')
user = pd.read_csv('user.csv')
user.head(2)
tourism_rating.head(2)
tourism_with_id.info()
tourism_with_id.isna().sum()
tourism_with_id.columns
#Remove the excess columns from tourism_with_id[]

tourism_with_id.drop(columns = ['Unnamed: 11', 'Unnamed: 12', 'Time_Minutes', 'Coordinate'], inplace = True)
tourism_with_id.info()
tourism_with_id.columns = tourism_with_id.columns.str.strip() 
#This is a string method that removes any leading and trailing whitespace from each column name.

#Create a bar plot and box plot to visualize the age distribution of the tourists visiting Indonesia.
import matplotlib.pyplot as plt

fig , (ax1,ax2) = plt.subplots(1,2, figsize = (12,6))

age_bins = pd.cut(user['Age'], bins=[17.978, 22.4, 26.8, 31.2, 35.6, 40.0])

age_counts = age_bins.value_counts().sort_index()

#categories = ['17.978,22.4', '22.4,26.8', '31.2,35.6', '35.6,40.0']
bar_width = 0.5

ax1.bar(age_counts.index.astype(str), age_counts.values, color='green',  width=bar_width)
fig.suptitle('Age Distribution', fontsize=20)

ax2.boxplot(user['Age'], patch_artist=True,vert=False, boxprops=dict(facecolor="orange"), widths=0.5, medianprops=dict(color='black'))
ax2.set_xlabel('Age')


plt.tight_layout()
plt.show()

'''
