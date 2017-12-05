# Text-Classification
Genre Classification of movies from movie reviews using scikit learn


1. Data Collection
    Getting the dataset from Facebook and Excel PowerQuery
    Using powerquery, we extract all the posts from the facebook page of the corresponding Movie along with the timstamp, userid and the
    genre of the movie and store it in an excel file in csv format.
    
2. Text extraction
    We use countVectorizer to vectorize all the documents (reviews).
    We form a bag of words with all the words in the training set.
    
3. Text Classification
    Using Multinomial classifier of Naive Bayes, we train the training set and then predict the genre of the test set using the classifier
