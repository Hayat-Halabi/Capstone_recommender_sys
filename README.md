# Capstone_recommender_sys

Problem Statement:
Indonesia wants to boost its tourism industry using advanced machine-learning techniques. You’re tasked with using the tourism data collected by the Indonesian government to understand tourist preferences and build a recommender system to recommend places to tourists.

Overview:
For effective marketing, it is of utmost importance to understand the customers(tourists) and their expectations. The recommender system is a great technique to augment the existing marketing outreach to prospects. This project requires you to perform exploratory data analysis and create a recommender system.

``` python
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
```
###
# 2. To understand the tourism highlights better, we should explore the data in depth.
###
# a. Explore the user groups used to get the tourism ratings.

The age distribution of users visiting the places and giving the ratings.
```python
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


user['city'] = user.Location.apply(lambda x: x.split(",")[0])
user1 = user.sort_values(by='city', ascending=True)

# Visualize the most frequented cities in Indonesia
import seaborn as sns
plt.figure(figsize = (15,10))

# user = user.sort_values(by=user.city, ascending=True)

sns.countplot(data = user1, y=user1['city'], palette='husl', orient='h',order=user['city'].value_counts().index, saturation=0.70)

plt.title('City Distribution')
plt.xlabel('count')
plt.ylabel('city')
           
plt.tight_layout()
plt.show()
```
###
# b. Next, explore the locations and categories of tourist spots.
###
What are the different categories of tourist spots?
Answer: There seems to be 6 different categories of tourist spots:

amusement parka
culture
nature preserve
the sea
place of worship
shopping center
``` python


tourism_with_id.Category = tourism_with_id.Category.str.strip().str.capitalize()
print(tourism_with_id.Category.unique())

tourism_with_id.City.unique()

#Visualize the number of visits for each tourism category to find the most popular category (type) of tourist spot.
plt.figure(figsize=(15,5))
ax = sns.countplot(data = tourism_with_id, x=tourism_with_id.Category, order =tourism_with_id['Category'].value_counts().index )
plt.title('Count of places across categories', fontsize = 16)
ax.bar_label(container=ax.containers[0], fmt='%d', fontsize=12, fontweight='bold')
plt.xlabel('Category', fontsize=16)
plt.ylabel('Count', fontsize = 14)
plt.show()
```
What kind of tourism each city/location is most famous or suitable for ?
Each location is famous for its own uniqueness.

Yogyakarta: For its Amusement Parks
Bandung: For its Nature preserve
Jakarta: for its Culture
Semarang: for its Nature preserve
Surabaya: for its Amusement parks and culture
``` python




#setting the colors to represent the graph values
color = ['seagreen', 'slateblue', 'darkred', 'saddlebrown']

tourism_with_id.City.unique()

tourism_with_id.Category

tourism_with_id = tourism_with_id.sort_values(by='Category', ascending=True)

# Visualize the distribution of the most famous category (type) of tourist spots in each city.
fig, axs = plt.subplots(1, 5, figsize=(20, 5))

tourism_with_id = tourism_with_id.sort_values(by='Category', ascending=True)

for idx, city in enumerate(tourism_with_id.City.unique()):
    sns.countplot(
        data=tourism_with_id[tourism_with_id.City == city], 
        ax=axs[idx], 
        palette=color, 
        x='Category', 
        width=bar_width, 
        order=tourism_with_id['Category'].value_counts().index
    )
    axs[idx].set_title(city, fontsize=20)
    axs[idx].set_xlabel('Category')
    axs[idx].set_ylabel('Count')
    axs[idx].set_yticks((0, 10, 20, 30, 40, 50, 60))
    axs[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
# Find the list of category types
tourism_with_id.Category.unique()

vc = tourism_with_id[tourism_with_id.Category == "Nature preserve"].City.value_counts()
# Plot the percentage distribution of tourist spots in each city. In this case, only consider spots categorized as "Nature preserve"

nature_preserve_data = tourism_with_id[tourism_with_id['Category'] == "Nature preserve"]

vc_percent = (vc / vc.sum()) * 100

plt.figure(figsize=(10, 6))
plt.pie(vc_percent, labels=vc_percent.index, autopct='%1.2f%%', startangle=140)
plt.title('Percentage Distribution of Nature Preserve Tourist Spots in Each City')
plt.axis('equal')
plt.show()

# Plot the price distribution for these tourist spots
tourism_with_id.Price
plt.boxplot(tourism_with_id["Price"])
plt.show()

tourism_with_id.Price.describe()
```
###
# c. To better understand the tourism ecosystem, we need to create a combined data with places and their ratings.
###
``` python

tourism_rating.head(2)
tourism_with_id.head(2)
```
###
# Calculate weighted average ratings for each place
###
```python 

# Calculate the weighted average of the 'Place_Ratings' column for each place/location.
weighted_avg = tourism_rating.groupby('Place_Id')['Place_Ratings'].mean().round(2).reset_index(name='Place_Ratings')
weighted_avg

# Merge this new average place rating to the tourism_with_id table. Hint: Join on the Place_Id column. Check the head of the new table to confirm the join operation.
place_ratings = tourism_with_id.merge(weighted_avg, on='Place_Id',sort=True)
place_ratings.head()
```
d. Use this data to figure out the spots that are most loved by the tourists.
Also, which city has the most loved tourist spots.
Solution : Picking up the places with average rating above 3.5 as most loved places and finding the cities where most of these highly rated spots are present

```python

# Merge this new average place rating to the tourism_with_id table. Hint: Join on the Place_Id column. Check the head of the new table to confirm the join operation.

place_ratings = tourism_with_id.merge(weighted_avg, on='Place_Id',sort=True)
place_ratings.head()

# Plot the percentage distribution of the cities with the most number of popular tourist spots. A popular tourist spot is defined as a place with an average rating greater than 3.5
popular = place_ratings[place_ratings['Place_Ratings']>3.5]

city_counts = popular['City'].value_counts()

popular_percent = (city_counts / city_counts.sum()) * 100

plt.figure(figsize=(10, 6))
plt.pie(popular_percent, labels=popular_percent.index, autopct='%1.2f%%', startangle=140)
plt.title('Cities with the Most Number of Popular Tourist Spots')
plt.axis('equal')
plt.show()

```
# Observations:

Record your observations here.
Based on the pie chart here are some of my observations:

Bandung: has the highest percentage of popular tourist spots, accounting for 36.67% of the total.
Yogyakarta: follows with 26.67% of the popular tourist spots.
Jakarta: is next, having 20.00% of the popular tourist spots.
Surabaya: has 13.33% of the popular tourist spots.
Semarang: has the smallest share, with only 3.33% of the popular tourist spots.
These percentages indicate the distribution of cities with the most popular tourist spots (defined as places with an average rating greater than 3.5) among the cities listed. Bandung, Yogyakarta, and Jakarta are the top three cities, making them key destinations for popular tourist attractions.

e. Indonesia provides a wide range of tourist spots ranging from historical and cultural beauties to advanced amusement parks. What category of places are users liking the most amongst these ?
Again picking up the places with average rating above 3.5 and finding out the which are the most liked categories

Most people liking the amusement parks very closely followed by the nature preserve.
```python

# Plot the distribution of the popular tourist spots (average ratings > 3.5) across the tourist categories
popular = place_ratings[place_ratings['Place_Ratings']>3.5]

plt.figure(figsize=(15,5))

ax1 = sns.countplot(data = popular, x=popular.Category, order =popular['Category'].value_counts().index )
plt.title('Count of places across categories', fontsize = 16)

ax1.bar_label(container=ax1.containers[0], fmt='%d', fontsize=12, fontweight='bold')
plt.xlabel('Category', fontsize=16)
plt.ylabel('Count', fontsize = 14)
plt.yticks(ticks=[0,2,4,6,8,10,12,14])
plt.show()

```
# Build a Recommendation model for the tourists
Create a dataframe with information about these spots to include place id, user rating, name, description, category, location and price.

Use the above data to develop a content based filtering model for recommendation. And use that to recommend other places to visit using the current tourist location(place name).
```python
# Create the dataframe for the recommender system.

DATA = tourism_with_id.drop(columns=['Lat','Long'])
recom_data = tourism_rating .merge(DATA, on ="Place_Id")
recom_data = recom_data.merge(user, on = 'User_Id')
recom_data.drop(columns=['Description', 'Category', 'Price','Location', 'Age','city'], inplace =True)
recom_data.head(2)

ratings_data = recom_data.groupby(['User_Id', 'Place_Name'])['Place_Ratings'].mean().unstack()
ratings_data

# Normalize user-item matrix

user_means = ratings_data.mean(axis=1)

data_norm = ratings_data.sub(user_means, axis=0)

data_norm.head(5)
# create a User similarity matrix using Pearson correlation

user_similarity= data_norm.T.corr(method='pearson')
user_similarity.head(5)
# Similarity
from sklearn.metrics.pairwise import cosine_similarity
user_similarity_cosine = cosine_similarity(data_norm.fillna(0))
user_similarity_cosine
# Pick a user ID
picked_userid = 1
# Remove picked user ID from the candidate list
user_similarity.drop(index=picked_userid, inplace=True)
# Take a look at the data
user_similarity.head()

# Number of similar users
n = 10

# User similarity threashold
user_similarity_threshold = 0.3

# picked user 
picked_userid = 1

# Get top n similar users
def get_top_n_similar_users(user_similarity, picked_userid, n=10, user_similarity_threshold=0.3):
    similarity_scores = user_similarity[picked_userid ]
    similar_users = similarity_scores[similarity_scores >= user_similarity_threshold]
    top_similar_users = similar_users.sort_values(ascending=False).head(n)
    return top_similar_users

top_similar_users = get_top_n_similar_users(user_similarity, picked_userid, n, user_similarity_threshold)

# Print out top n similar users

print(f"The similar users for user {picked_userid} are:")
print(top_similar_users)


# List the places that the target user has visited and rated
def list_place_and_rate(rating,user):
    name = rating[rating.index==user].dropna(axis = 1)
    return name

user_rated_item = list_place_and_rate(data_norm,picked_userid )
user_rated_item.T

# List the places that similar users visited and rated.
def places_similar_users_rated(data, top_similar_users):
    result = pd.DataFrame()
    for user in top_similar_users.index:
        user_data = data.loc[user].dropna()
        result = pd.concat([result, user_data], axis=1)
    result = result.sort_index()
    return result

users_rated_item = places_similar_users_rated(data_norm, top_similar_users)
users_rated_item.T.sort_index()

# Remove the places already visitied
# Take a look at the data

user = list(user_rated_item.keys())
user
users = users_rated_item.T

users_to_drop = []

for item in user:
    if item in users.keys():
        users_to_drop.append(item)

print (len(users_to_drop))
new_df = users.drop(users_to_drop, axis = 1).sort_index()
new_df

# A dictionary to store item scores
# Convert dictionary to pandas dataframe
# Sort the places by score
# Display top m places

item_scores = {}
import numpy as np
for place in new_df.columns:
    ratings = []
    users = []
    for user in new_df.index:
        if not pd.isna(new_df.at[user, place]):
            ratings.append(new_df.at[user, place])
            users.append(user)

    average_rating = np.mean(ratings) if ratings else 0
    item_scores[place] = {
        'average_rating': average_rating,
        'users': users
    }

sorted_places = sorted(item_scores.items(), key=lambda x: x[1]['average_rating'], reverse=True)

top_m = 10
top_places = sorted_places[:top_m]
df = pd.DataFrame(top_places)
place_names = [place for place, details in top_places]
place_names

```
The observation from my results indicates the average ratings and the number of users who rated each place

Masjid Nasional Al-Akbar and Tektona Waterpark have the highest average rating (2.418) and are rated by user 136.

Kebun Tanaman Obat Sari Alam has an average rating of 2.397 and is rated by user 90.

Glamping Lakeside Rancabali has an average rating of 2.297 and is rated by user 155.

Museum Fatahillah and Rainbow Garden have the same average rating of 2.297, rated by user 155.

Upside Down World Bandung has an average rating of 2.058 and is rated by user 89.

Ciputra Waterpark and Pelabuhan Marina have the same average rating of 2.042, rated by user 124.

Kawasan Malioboro has the lowest average rating of 2.017 and is rated by users 86 and 124.
###
# Key Observations:
###
Ratings Consistency: Some places have identical ratings and user counts, indicating the same user rated multiple places with the same score. For example, Masjid Nasional Al-Akbar and Tektona Waterpark both have ratings from user 136.

User Engagement: Users 136, 155, and 124 are quite active, rating multiple places.

Diverse Ratings: Some places like Kawasan Malioboro have ratings from multiple users (86 and 124), while others have ratings from a single user.

Top-rated Places: The highest average rating is 2.418, shared by two places, suggesting a tie for the most favored location among the users.
###
# Insights from graphs
###
Indonesia offers a diverse range of tourist spots, but the popularity varies greatly among different categories. While amusement parks and nature preserves are well-liked, efforts could be made to enhance the appeal of less popular categories to attract a wider audience.

The bargraphs ploted above shows that tourists prefer engaging and interactive experiences like amusement parks and nature preserves over more passive or niche experiences like visiting the sea or places of worship.

Amusement Parks are the Most Liked: Amusement parks are the most liked category, users highly favor amusement parks in Indonesia, making them a popular choice among tourists.
Nature Preserves are also Highly Favored: Nature preserves come second, there is strong interest in natural attractions, indicating that tourists appreciate the beauty and experience offered by nature preserves.
Cultural Attractions Hold Significant Interest: The culture category has 6 places with ratings above 3.5, While not as high as amusement parks and nature preserves, cultural attractions still attract a significant number of tourists, reflecting an interest in historical and cultural beauty.
Low Interest in the Sea and Places of Worship: The sea and places of worship each have only 1 place with an average rating above 3.5. This indicates that these categories are less favored by tourists, possibly due to limited options or lower overall appeal.
The bargraphs ploted above shows that tourists prefer engaging and interactive experiences like amusement parks and nature preserves over more passive or niche experiences like visiting the sea or places of worship.
```
```
# Link to access data used above:
```
https://drive.google.com/file/d/16wm8OJmtqpMs3gP4wn4rdcTMk8n4PbzJ/view?usp=drive_link
```
```
https://docs.google.com/spreadsheets/d/17qC22hG_KowrydRyWL8mk078xusvGoS9/edit?usp=drive_link&ouid=106027145200045519765&rtpof=true&sd=true
```
```
https://drive.google.com/file/d/1wGWATStnULbWGnhy4pN_XYxacmonrip9/view?usp=drive_link
