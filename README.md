# Recommender Systems - Movie Recommendation Project

## Outline of the Project

In the beginning of 2006, Netflix, then a service peddling discs of every movie and TV show under the sun, announced “The Netflix Prize”, a competition that lured many engineers and programmers.  The mission was to make the company’s existing recommendation engine 10% more accurate.  The reward for solving this was one million dollars. Word of the competition immediately spread like a virus through computer science circles, technology bloggers, research communities, and even the mainstream media. And while a million dollars created attention, it was the data set -- over 100 million ratings of 17,770 movies from 480,189 customers -- that had number-crunching nuts salivating. There was nothing like it at the time and there hasn't been anything quite like it since. Following the announcement, 30,000 Netflix enthusiasts downloaded the information and set out to unlock this secret of the recommendation algorithm.  On June 26, 2009, a team called “BellKor’s Pragmatic Chaos” achieved a 10.5% improvement on Netflix’s recommending engine with Root Mean Squared Error known as RMSE of 0.8558. A month later another team “ Ensemble” achieved 10.09% RMSE over the Netflix’s and in September 2009, Netflix announced the BellKor’s team as the winner.  

Recommendation algorithms are at the core of the Netflix products and many online retailers such as Amazon, Zara, Guardian, and many others. They provide customers with personalised suggestions to reduce the amount of time and frustration to find something great content to watch or buy. Because of the importance of the recommendations, many companies continually seek to improve them by advancing the state-of-the-art in the field. 

In this project I will be building a movie recommendation algorithm using the famous big dataset “MovieLens” dataset and Apache Spark utilities to handle the dataset and building the recommendation algorithm.  This is a fascinating project which scales well to big data and has many interesting applications and variety of ways to solve the problem and this is why I have chosen this project.  

### Solution 

Assuming that rating matrix with mxn dimension where m is the user (500k) and n is the item (17k movies), this matrix would be extremely sparse with only 100 million ratings and remaining 8.4 billion ratings missing – about 99% of possible ratings – because users only rate a small portion of the movies.  The goal of the recommender algorithm is to predict those missing ratings so that user can be recommended of items of similar ratings or users items of similar to their tastes.  Given this very large matrix, the only feasible that competitors attempt this by performing dimensionality reduction which was the basis for the winning algorithm and can be done via matrix factorization.  Matrix factorization many advantages; when explicit feedback is not available such as ratings which most of the application don’t, an inference can be used using implicit feedback which indirectly reflects the user’s opinion by observing their behaviour including purchase history, browsing history, search patterns or even mouse movements.  



### Types of Recommender Systems 

Two most ubiquitous types of recommender systems are Content-Based and Collaborative Filtering (CF). Collaborative filtering produces recommendations based on the knowledge of users’ attitude to items, that is it uses the “wisdom of the crowd” to recommend items. In contrast, content-based recommender systems focus on the attributes of the items and give you recommendations based on the similarity between them. In general, Collaborative filtering (CF) is the workhorse of recommender engines. The algorithm has a very interesting property of being able to do feature learning on its own, which means that it can start to learn for itself what features to use. CF can be divided into Memory-Based Collaborative Filtering and Model-Based Collaborative filtering. 

**Memory-Based Collaborative Filtering** - Memory-Based Collaborative Filtering approaches can be divided into two main sections: user-item filtering and item-item filtering. A user-item filtering will take a particular user, find users that are similar to that user based on similarity of ratings, and recommend items that those similar users liked. In contrast, item-item filtering will take an item, find users who liked that item, and find other items that those users or similar users also liked. It takes items and outputs other items as recommendations. A distance metric commonly used in recommender systems is cosine similarity, where the ratings are seen as vectors in n-dimensional space and the similarity is calculated based on the angle between these vectors. As these are memory based algorithms and easy to implement, they do not scale well to real world scenarios and does address the well known cold start problem.  

**Model-based Collaborative Filtering** on another hand is based on matrix factorization (MF) which has received greater exposure, mainly as an unsupervised learning method for latent variable decomposition and dimensionality reduction. Matrix factorization is widely used for recommender systems where it can deal better with scalability and sparsity than Memory-based CF. The goal of MF is to learn the latent preferences of users and the latent attributes of items from known ratings (learn features that describe the characteristics of ratings) to then predict the unknown ratings through the dot product of the latent features of users and items. When you have a very sparse matrix, with a lot of dimensions, by doing matrix factorization you can restructure the user-item matrix into low-rank structure, and you can represent the matrix by the multiplication of two low-rank matrices, where the rows contain the latent vector. You fit this matrix to approximate your original matrix, as closely as possible, by multiplying the low-rank matrices together, which fills in the entries missing in the original matrix.

### SVD

A well-known matrix factorization method is Singular value decomposition (SVD). Collaborative Filtering can be formulated by approximating a matrix X by using singular value decomposition. The winning team at the Netflix Prize competition used SVD matrix factorization models to produce product recommendations. Matrix X can be factorized to U, S and V. The U matrix represents the feature vectors corresponding to the users in the hidden feature space and the V matrix represents the feature vectors corresponding to the items in the hidden feature space.
Dataset 

The MovieLens dataset were collected by GroupLens Research Project at the university of Minnesota.  Dataset consists of 100,000 ratings (1-5) from 943 users on 1682 movies where each user has rated at least 20 movies and simple demographic info for the users. The data was collected through the MovieLens web site (movielens.umn.edu) during the seven-month period from September 19th, 1997 through April 22nd, 1998. This data has been cleaned up. I am using this dataset because its sufficiently large dataset to address the question of big data and applicable to this project which is building Movie Recommendation algorithm.  

My hypothesis is that this dataset is normal under the circumstances and fits the distribution of similar datasets in the real world.  

### Evaluating of the Model

There are many evaluation metrics but one of the most popular metric used to evaluate accuracy of predicted ratings is Root Mean Squared Error (RMSE). I will therefore be using this to evaluate the model between the ground truth of test set ratings column and predicted ratings.  

### Technologies 

I will be using Hadoop file system to load the dataset from the <a href="http://files.grouplens.org/datasets/movielens/ml-100k.zip">grouplens.org</a> website using the command line interface once loaded I will then run a pyspark session on Jupyter-notebook to load the data from Hadoop. I will then be performing some cleaning of the data on Spark RDD/DataFrame following by some statistical analysis of the dataset then preparation for Spark ML for the final machine learning algorithm and prediction.   

### Approach to the Project 

I will be taking the following steps to this project: 

1.	Import dependencies and start a spark session
2.  Load the dataset and merge with the user information
3.	Perform some explanatory analysis of the dataset 
4.	Prepare the dataset for machine learning including train/test split and feature preparation 
5.	Train a collaborative filtering model using Alternating least squares (ALS)
6.	Test the model 
7.	Evaluate the model using RMSE
8.  Tune Hyper Parameters
8.	Finally recommend movies to users 
9.	Conclusions 
10. Referrences 


### 1. Import Dependencies 


```python
#import spark
from pyspark.sql import SparkSession
import numpy as np
import matplotlib.pyplot as plt
import pyspark
from pyspark.context import SparkContext
from pyspark.sql.functions import sum, col, desc, mean, count, round
from pyspark.sql.types import *
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
import warnings
```


```python
# Start a Spark Session
app_name = 'Recommender Systems in Spark'
spark = SparkSession\
    .builder \
    .appName(app_name)\
    .config('spark.some.config.option', 'some_value')\
    .getOrCreate()
```

### 2. Load the datasets 


```python
#load the main rating dataset as RDD
lines = spark.read.text("ml-100k/u.data").rdd
```


```python
#load the movie id dataset as RDD
lines1 = spark.read.text("ml-100k/u.item").rdd
```


```python
# split the cols by seperator 
parts = lines.map(lambda row: row.value.split("\t"))
```


```python
# split the cols by seperator 
parts1 = lines1.map(lambda row: row.value.split("|"))
```


```python
# extract only the first two columns of the RDD
parts1 = parts1.map(lambda col: [col[i] for i in [0,1]])
```


```python
# name the columns and change the dtypes to appropriete
ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),rating=float(p[2]), timestamp=int(p[3])))
```


```python
# name the columns and change the dtypes to appropriete
movieRDD = parts1.map(lambda p: Row(movieId=int(p[0]), movieName=str(p[1])))
```


```python
# create a DF from the spark ratings RDD
ratings = spark.createDataFrame(ratingsRDD)
```


```python
# create a DF from the spark movie RDD
movie = spark.createDataFrame(movieRDD)
```


```python
#join ratings to movie dataframe
ratings_table = ratings.alias('a').join(movie.alias('b'), ratings['movieId'] == movie['movieId'], 'left').select("a.userId", "a.movieId", "a.rating", "b.movieName")
```

### 3. Exploratory Analysis


```python
# show the total number of rows in the ratings dataset
ratings_table.count()
```




    100000




```python
# show the total number of columns 
ratings_table.columns
```




    ['userId', 'movieId', 'rating', 'movieName']




```python
# show the top 20 lines 
ratings_table.show(20)
```

    +------+-------+------+--------------------+
    |userId|movieId|rating|           movieName|
    +------+-------+------+--------------------+
    |   138|     26|   5.0|Brothers McMullen...|
    |   224|     26|   3.0|Brothers McMullen...|
    |    18|     26|   4.0|Brothers McMullen...|
    |   222|     26|   3.0|Brothers McMullen...|
    |    43|     26|   5.0|Brothers McMullen...|
    |   201|     26|   4.0|Brothers McMullen...|
    |   299|     26|   4.0|Brothers McMullen...|
    |    95|     26|   3.0|Brothers McMullen...|
    |    89|     26|   3.0|Brothers McMullen...|
    |   361|     26|   3.0|Brothers McMullen...|
    |   194|     26|   3.0|Brothers McMullen...|
    |   391|     26|   5.0|Brothers McMullen...|
    |   345|     26|   3.0|Brothers McMullen...|
    |   303|     26|   4.0|Brothers McMullen...|
    |   401|     26|   3.0|Brothers McMullen...|
    |   429|     26|   3.0|Brothers McMullen...|
    |   293|     26|   3.0|Brothers McMullen...|
    |   270|     26|   5.0|Brothers McMullen...|
    |   442|     26|   3.0|Brothers McMullen...|
    |   342|     26|   2.0|Brothers McMullen...|
    +------+-------+------+--------------------+
    only showing top 20 rows
    



```python
# show the number of unique users 
ratings_table.select('UserId').distinct().count()
```




    943




```python
# show the number of unique movies 
ratings_table.select('MovieId').distinct().count()
```




    1682




```python
# show the average average ratings by movieID and Count of Raters where the number of Users is greater than 10
ratings_table.groupBy("MovieId").agg(round(mean("rating"),0).alias("RatingAverage"),count('UserId').alias("NumberofUsers")).filter(col("NumberofUsers") > 10).sort(desc("RatingAverage")).show(20)
```

    +-------+-------------+-------------+
    |MovieId|RatingAverage|NumberofUsers|
    +-------+-------------+-------------+
    |    237|          4.0|          384|
    |    847|          4.0|           55|
    |    241|          4.0|          128|
    |    705|          4.0|          137|
    |    287|          4.0|           78|
    |    502|          4.0|           57|
    |    736|          4.0|           78|
    |    191|          4.0|          276|
    |    474|          4.0|          194|
    |     19|          4.0|           69|
    |    558|          4.0|           70|
    |     65|          4.0|          115|
    |    656|          4.0|           44|
    |    222|          4.0|          365|
    |    385|          4.0|          208|
    |    730|          4.0|           24|
    |    270|          4.0|          136|
    |    293|          4.0|          147|
    |    418|          4.0|          129|
    |    965|          4.0|           21|
    +-------+-------------+-------------+
    only showing top 20 rows
    



```python
#movies with the highest ratings
ratings_table.groupby('movieName').agg(count('userId').alias('NumberofVotes')).sort(desc("NumberofVotes")).show(20)
```

    +--------------------+-------------+
    |           movieName|NumberofVotes|
    +--------------------+-------------+
    |    Star Wars (1977)|          583|
    |      Contact (1997)|          509|
    |        Fargo (1996)|          508|
    |Return of the Jed...|          507|
    |    Liar Liar (1997)|          485|
    |English Patient, ...|          481|
    |       Scream (1996)|          478|
    |    Toy Story (1995)|          452|
    |Air Force One (1997)|          431|
    |Independence Day ...|          429|
    |Raiders of the Lo...|          420|
    |Godfather, The (1...|          413|
    | Pulp Fiction (1994)|          394|
    |Twelve Monkeys (1...|          392|
    |Silence of the La...|          390|
    |Jerry Maguire (1996)|          384|
    |  Chasing Amy (1997)|          379|
    |    Rock, The (1996)|          378|
    |Empire Strikes Ba...|          367|
    |Star Trek: First ...|          365|
    +--------------------+-------------+
    only showing top 20 rows
    



```python
ratings_table.groupby('movieName').agg(round(mean('rating'),0).alias('AverageRatings')).sort(desc("AverageRatings")).show(20)
```

    +--------------------+--------------+
    |           movieName|AverageRatings|
    +--------------------+--------------+
    |Maya Lin: A Stron...|           5.0|
    |Pather Panchali (...|           5.0|
    |Entertaining Ange...|           5.0|
    |         Anna (1996)|           5.0|
    |     Star Kid (1997)|           5.0|
    |      Everest (1998)|           5.0|
    |Some Mother's Son...|           5.0|
    |They Made Me a Cr...|           5.0|
    |Santa with Muscle...|           5.0|
    |Aiqing wansui (1994)|           5.0|
    |Marlene Dietrich:...|           5.0|
    |Someone Else's Am...|           5.0|
    |Saint of Fort Was...|           5.0|
    |Great Day in Harl...|           5.0|
    |  Prefontaine (1997)|           5.0|
    |North by Northwes...|           4.0|
    |When We Were King...|           4.0|
    |Heavenly Creature...|           4.0|
    |    Notorious (1946)|           4.0|
    |       Psycho (1960)|           4.0|
    +--------------------+--------------+
    only showing top 20 rows
    



```python
#show statistics 
ratings_table.describe().show()
```

    +-------+------------------+-----------------+-----------------+--------------------+
    |summary|            userId|          movieId|           rating|           movieName|
    +-------+------------------+-----------------+-----------------+--------------------+
    |  count|            100000|           100000|           100000|              100000|
    |   mean|         462.48475|        425.53013|          3.52986|                null|
    | stddev|266.61442012750865|330.7983563255848|1.125673599144316|                null|
    |    min|                 1|                1|              1.0|'Til There Was Yo...|
    |    max|               943|             1682|              5.0|� k�ldum klaka (C...|
    +-------+------------------+-----------------+-----------------+--------------------+
    


### 4. Prepare for modelling 


```python
#split dataset for train/test split
(training, test) = ratings_table.randomSplit([0.8, 0.2])
```

### 5. Train Model - ALS Collaborative Filtering 


```python
#train ALS model with initial params
als = ALS(maxIter=10, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",coldStartStrategy="drop")
model = als.fit(training)
```

### 6. Test Model 


```python
predictions = model.transform(test)
```


```python
predictions.show()
```

    +------+-------+------+--------------------+----------+
    |userId|movieId|rating|           movieName|prediction|
    +------+-------+------+--------------------+----------+
    |   251|    148|   2.0|Ghost and the Dar...| 2.8130546|
    |    26|    148|   3.0|Ghost and the Dar...| 2.7410953|
    |   332|    148|   5.0|Ghost and the Dar...| 4.0903797|
    |   916|    148|   2.0|Ghost and the Dar...| 2.1030276|
    |   236|    148|   4.0|Ghost and the Dar...| 4.1956735|
    |   602|    148|   4.0|Ghost and the Dar...| 3.7098408|
    |   663|    148|   4.0|Ghost and the Dar...| 3.1607192|
    |   727|    148|   2.0|Ghost and the Dar...| 3.3720503|
    |   190|    148|   4.0|Ghost and the Dar...|  3.421185|
    |   363|    148|   3.0|Ghost and the Dar...| 2.7118642|
    |   308|    148|   3.0|Ghost and the Dar...|  2.647034|
    |   479|    148|   2.0|Ghost and the Dar...| 3.2181206|
    |   455|    148|   3.0|Ghost and the Dar...|   3.13608|
    |   891|    148|   5.0|Ghost and the Dar...| 3.1954665|
    |   552|    148|   3.0|Ghost and the Dar...| 3.0373685|
    |   870|    148|   2.0|Ghost and the Dar...| 2.2289264|
    |   733|    148|   3.0|Ghost and the Dar...| 1.2470127|
    |    59|    148|   3.0|Ghost and the Dar...| 2.9585698|
    |   717|    148|   3.0|Ghost and the Dar...| 3.7365098|
    |   757|    148|   4.0|Ghost and the Dar...| 3.0711873|
    +------+-------+------+--------------------+----------+
    only showing top 20 rows
    


The ratings predictions for the majority of the top 20 rows are pretty close however some are way off. We can improve on this by changing the parameters later. 

### 7. Evaluate model


```python
#evaluate using RMSE 
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")

rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))
```

    Root-mean-square error = 1.1016716161905122


### 8. Improve the model 


```python
# define a grid search function to tune the best parameters. 
```


```python
def tune_ALS(train_data, validation_data, maxIter, regParams, ranks):
    """
    grid search function to select the best model based on RMSE of
    validation data
    Parameters
    ----------
    train_data: spark DF with columns ['userId', 'movieId', 'rating']
    
    validation_data: spark DF with columns ['userId', 'movieId', 'rating']
    
    maxIter: int, max number of learning iterations
    
    regParams: list of float, one dimension of hyper-param tuning grid
    
    ranks: list of float, one dimension of hyper-param tuning grid
    
    Return
    ------
    The best fitted ALS model with lowest RMSE score on validation data
    """
    # initial
    min_error = float('inf')
    best_rank = -1
    best_regularization = 0
    best_model = None
    for rank in ranks:
        for reg in regParams:
            # get ALS model
            als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating",coldStartStrategy="drop").setMaxIter(maxIter).setRank(rank).setRegParam(reg)
            # train ALS model
            model = als.fit(train_data)
            # evaluate the model by computing the RMSE on the validation data
            predictions = model.transform(validation_data)
            evaluator = RegressionEvaluator(metricName="rmse",
                                            labelCol="rating",
                                            predictionCol="prediction")
            rmse = evaluator.evaluate(predictions)
            print('{} latent factors and regularization = {}: '
                  'validation RMSE is {}'.format(rank, reg, rmse))
            if rmse < min_error:
                min_error = rmse
                best_rank = rank
                best_regularization = reg
                best_model = model
    print('\nThe best model has {} latent factors and '
          'regularization = {}'.format(best_rank, best_regularization))
    return best_model
```


```python
tune_ALS(training, test, 10, [0.1,0.5,0.9], [15,20,25])
```

    15 latent factors and regularization = 0.1: validation RMSE is 0.9287849273255995
    15 latent factors and regularization = 0.5: validation RMSE is 1.0715229680513707
    15 latent factors and regularization = 0.9: validation RMSE is 1.3087345398118395
    20 latent factors and regularization = 0.1: validation RMSE is 0.9304680790220787
    20 latent factors and regularization = 0.5: validation RMSE is 1.0715109662450097
    20 latent factors and regularization = 0.9: validation RMSE is 1.3087348115726882
    25 latent factors and regularization = 0.1: validation RMSE is 0.9290379823458554
    25 latent factors and regularization = 0.5: validation RMSE is 1.0712263684125654
    25 latent factors and regularization = 0.9: validation RMSE is 1.3087365494830216
    
    The best model has 15 latent factors and regularization = 0.1





    ALSModel: uid=ALS_c940631cd6a1, rank=15




```python
# try the new hyper parameters
als = ALS(maxIter= 10, regParam=0.1, rank = 15, userCol="userId", itemCol="movieId", ratingCol="rating",coldStartStrategy="drop")
model = als.fit(training)

predictions = model.transform(test)

#evaluate using RMSE 
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")

rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))
```

    Root-mean-square error = 0.9287849273255995


### 9. Recommend movies to Users


```python
#change the movies df to Pandas
movies = movie.toPandas()
```


```python
warnings. filterwarnings('ignore')
```


```python
def recommend_to_user(user_id, model, num_recommendations = 5):
    
    """Recommend to user based on similar users and return the top movie names"""
    
    # Generate top 5 movie recommendations for each user
    userRecs = model.recommendForAllUsers(num_recommendations)
    
    #filter for user 471
    recommendations = userRecs.filter(userRecs.userId == user_id).toPandas()
    recommendations = recommendations['recommendations']
    
    #extract movieId and score 
    rec_list = [v[0] for _ in recommendations for v in _]
    rating_list = [v[1] for _ in recommendations for v in _]
    
    #Extract the movie title and match to movie Id
    recommended_list = movies[movies.movieId.isin(rec_list)]
    recommended_list.loc[:,'value'] = rating_list
    recommended_list.sort_values('value',ascending = False)
    
    return recommended_list[['movieName']].values.tolist()
```


```python
# try user number 471
recommend_to_user(471, model, num_recommendations = 10)
```




    [['Rocket Man (1997)'],
     ['Turbulence (1997)'],
     ['In Love and War (1996)'],
     ['Rendezvous in Paris (Rendez-vous de Paris, Les) (1995)'],
     ['Love! Valour! Compassion! (1997)'],
     ['That Old Feeling (1997)'],
     ['Star Maps (1997)'],
     ['Tom and Huck (1995)'],
     ['Hurricane Streets (1998)'],
     ['Angel Baby (1995)']]




```python
# also for user id number 251 and recommend top 5
recommend_to_user(251, model, num_recommendations = 5)
```




    [['Godfather, The (1972)'],
     ['Rear Window (1954)'],
     ['Mina Tannenbaum (1994)'],
     ['Boys, Les (1997)'],
     ['Angel Baby (1995)']]




```python
spark.stop()
```

### 10. Conclusion

So far in this notebook, i have built a collaborative filtering recommender system with matrix factorization. Matrix factorization is great for solving many real world problem as i have touched on earlier.  The issues of recommender systems boil down to three main issues:

1. Limiations and scalabity issue - with Matrix factorization; i have demonstrated that this solution can easily scale to large datasets and big data. 
2. Popularity bias - this refers to when the system recommends the movies with the most interactions without any personalisation and matrix. Because ALS model learns to factorize rating matrix into user and movie representation, it allows model to predict better personalised movie ratings for users. 
3. item cold-start problem - this is when when movies added to the catalogue have either none or very little interactions while recommender rely on the movie’s interactions to make recommendations although this is also one of its disadvantages as the matrix factorization technique may suffer from cold start problem in not being able to serve a recommendation for new users that have bought nothing. 

Because the ability of Matrix factorization to deal with all the above issues, it was the winner for the Netflix Prize competition.


### 11. Referrences 

Books <br>
Minning of Massive Datasets (Third Addition) by Jure Leskovec & Anand Rajaraman and etal <br>
Big Data Processing with Apache Spark by Manuel Ignacio Franco <br.
Learning PySpark by Tomasz Drabas & Danny Lee <br>

Web <br>
<a href = 'https://en.wikipedia.org/wiki/Netflix_Prize'>Wikipedia </a> <br>
<a href = 'https://www.thrillist.com/entertainment/nation/the-netflix-prize'> Netflix Blog </a> <br>
<a href = 'https://towardsdatascience.com/prototyping-a-recommender-system-step-by-step-part-1-knn-item-based-collaborative-filtering-637969614ea'> TowardsDataScience </a>
