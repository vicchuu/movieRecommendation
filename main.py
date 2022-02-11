#""""Movie recommendation system...""""
import pandas as ps

from sklearn.feature_extraction.text import TfidfVectorizer

dataset=ps.read_csv("movies.csv")
dataset['genres']=dataset['genres'].replace([' '],',')

print(dataset['genres'])
#print(dataset.info())

neededColums=["genres","keywords","tagline","cast","crew"]



"""preprocessing for filling NA value"""
for selectedColumn in neededColums:
    dataset[selectedColumn]=dataset[selectedColumn].fillna(" ")


"""new dataset"""
newDataSet=dataset["genres"]+dataset["keywords"]+dataset["tagline"]+dataset["cast"]+dataset["crew"]


"""For printing values which has more revenue  """
#
# revenueDataset=list(enumerate(dataset["revenue"]//dataset["budget"]))
#
# for key,value in (revenueDataset):
#     if value<0:
#         revenueDataset[key]=0.0
#
#
# dataset["revenuePercent"]=revenueDataset
#
# """removing INF revenue value in """
#
# dataset=dataset.sort_values(by="revenuePercent",ascending=False)
#
# #print(dataset.head())
#
# for a in range(100):
#     print(a," :",dataset.iloc[a]["title"] ," -Release Date :",dataset.iloc[a]["release_date"]," -Revenue % :",dataset.iloc[a]["revenuePercent"][0])
#
# #for key , value in range(100):
# #    print("",a," Movie Name :",dataset["title"]," Revenue in %:",dataset["revenuePercent"].where(dataset.iloc[a]))
# #for a in
# #print("revenue :",revenueDataset)
#
# #print(newDataSet.isnull().sum())
# #print(newDataSet.shape)

"""#CREATING AN instanve for tfidvectorizer from sklearn.featureextraction.text import TFID"""
vector1=TfidfVectorizer()


"""fit into vectorizer"""


featureVector=vector1.fit_transform(newDataSet)
#print(featureVector)

"""First find similarity between each movies by similaruty score and need to compare """

from sklearn.metrics.pairwise import  cosine_similarity

similar=cosine_similarity(featureVector)

import streamlit as st

st.title("#Movie recommendation  based on user Input ")

st.write(dataset.genres.unique())
option = st.selectbox(
          'Select Genre type !',
            ('Action','Romantic','Drama'))

st.write('You selected:', option)
#st.selectbox(label="Select any genre :",options=)
movieName="action"#input("Please enter your interested Generes  :")#"SpiderMan"

"""Creating a list based on all title to compare and check"""

movieList=((dataset["title"])).to_list()

print(movieList[0])


import difflib as df

sameName=df.get_close_matches(movieName,movieList)

print(sameName) # spiderman , spiderman 3 , spiderMan3 ordered based on most suited one

"""Exact matches"""

sameExactname=sameName[0] #chooses first index value


"""Find exact index value of above movie"""

indexValueOfMovie=dataset[dataset.title==sameExactname]["index"].values[0]
print(indexValueOfMovie)

"""Now we find out index of exact matching movie name and data. Noe compare exact data with all movie data"""

matchedSimilarity=list(enumerate(similar[indexValueOfMovie]))


# print(matchedSimilarity)


"""sorting these movies in descending order (realted to those exact movies)"""

matchedSimilarityOrder=sorted(matchedSimilarity,key=lambda x:x[1],reverse=True)

"""now printing recommended movies"""


for key in range(10):

     index=matchedSimilarityOrder[key][0]
     title=dataset[dataset.index==index]["title"].values[0]
     rating =dataset["vote_average"][index]
     if rating>5.0:
        st.write(key," . ",title , " rating :", rating)