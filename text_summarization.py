import nltk
import sklearn
import numpy
import pandas as pd
import duckdb
import html
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression









articles = pd.read_csv('news.csv')

df = duckdb.sql('SELECT article, highlights as summary FROM articles').df()

#creating a bunch of articles where each article is tokenized into sentences
articles = [nltk.sent_tokenize(text) for text in df['article']]





def clean_sentence(sentence):
    cleaned_sentence = sentence.lower()
    cleaned_sentence = re.sub(r"[^A-Za-z0-9\s'\-]", " ", cleaned_sentence)
    cleaned_sentence = re.sub(r"\s+", " ", cleaned_sentence).strip()   # collapse spaces
    return cleaned_sentence



cleaned_articles = []
for article in articles:
    cleaned_article = []
    for sentence in article:
        cleaned_sentence = clean_sentence(sentence)
        if cleaned_sentence:
            cleaned_article.append(cleaned_sentence)  
    cleaned_articles.append(cleaned_article)


train_articles, test_articles, train_summaries, test_summaries = train_test_split(
    cleaned_articles, df['summary'], test_size=0.33, random_state=42)



vec_TfIDF = TfidfVectorizer(
    ngram_range= (1,2),
    min_df= 2,
    max_df= 0.5,
    sublinear_tf=True, #good for long documents: use 1+log(tf) where there is diminishing returns for repeats
    token_pattern= r"(?u)\b\w[\w'-]*\b"
    
)

# a list of all the sentences in the training set
flattened_train = [sentence for article in train_articles for sentence in article]
vec_TfIDF.fit(flattened_train)

all_words = [sentence.split() for sentence in flattened_train]
model = Word2Vec(all_words, min_count=1,vector_size= 300)


def create_overlap_labels(article, summary):
    """
    Creates a list of labels, 0 or 1, based on the overlap score between a specific sentence and the summary
    """
    overlap_labels = []
    sentences = [sentence for sentence in article]
    cleaned_summary = clean_sentence(summary)
    summary_words = cleaned_summary.split()

    #creates a set of all of the unique summary words, eliminating repeats
    unique_summary_words = set(summary_words)
    for sentence in sentences:
        sentence_words = sentence.split()
        unique_sentence_words = set(sentence_words)
        words_in_summary_and_sentence = 0
        for word in unique_sentence_words:
            if word in unique_summary_words:
                words_in_summary_and_sentence += 1
        overlap = words_in_summary_and_sentence/len(unique_sentence_words)
        if overlap >= 0.5:
            overlap_labels.append(1)
        else:
            overlap_labels.append(0)
    return overlap_labels

#list of 0s and 1s across all sentences 
y_labels = []
for article,summary in zip(train_articles, train_summaries):
    overlap_labels = create_overlap_labels(article,summary)
    y_labels.extend(overlap_labels)

def add_extra_features(article):
    """
    Given an article, add extra features to each sentence of each article 
    """
    extra_features = []
    
    #for sentence in each article
    for i in range(len(article)):
        if len(article) > 1:
                #position of sentence in article
            position = i/(len(article) - 1)
        else:
             position = 0.0
            #number of words
        length = len(article[i].split())
        length = np.log1p(length)
        extra_features.append([position, length])
    return extra_features

#since extra features is a nested list, it will create a nested list. 
# This is so I have the same dimensionality as my tfIDF vector
total_extra_features = []
for article in train_articles:
    extra_features = add_extra_features(article)
    total_extra_features.extend(extra_features)

total_extra_features = np.array(total_extra_features)
extra_features_sparse = csr_matrix(total_extra_features)    
combined_train = hstack([vec_TfIDF.transform(flattened_train), extra_features_sparse])



model2 = LogisticRegression(
    penalty='l2',
    class_weight='balanced',  # helps if many more 0s than 1s
    max_iter=1000,
    solver='liblinear')
model2.fit(combined_train, y_labels)


def logistic_classification_summary(article, num_summary_sentences):
    transformed_article = vec_TfIDF.transform(article)
    extra_features = np.array(add_extra_features(article))
    extra_features_sparse = csr_matrix(extra_features)    
    combined_article = hstack([transformed_article, extra_features_sparse])
    probabilities = model2.predict_proba(combined_article)[:,1]
    ranked_indexes = np.argsort(-probabilities)
    top_indexes = ranked_indexes[:num_summary_sentences]
    top_indexes_sorted = sorted(top_indexes)
    summary_sentences = [article[i] for i in top_indexes_sorted]
    return ". ".join(summary_sentences)



def kmeans_summary(article):
    sentence_vector = []
    #this is turning each sentence into a point, so it can be plotted onto a graph
    for sentence in article:
        sentence = clean_sentence(sentence)
        tokens = sentence.split()

        # Keep only words that are in the Word2Vec vocabulary
        tokens = [w for w in tokens if w in model.wv.key_to_index]

        if not tokens:
            # If no tokens are known, use a zero vector for this sentence
            sent_vec = np.zeros(model.vector_size, dtype=float)
        else:
            # Average the word vectors for the known tokens
            word_vecs = [model.wv[w] for w in tokens]        # list of (300,) arrays
            sent_vec = np.mean(word_vecs, axis=0)            # (300,)

        sentence_vector.append(sent_vec)

    # Convert to a proper 2D array: shape (n_sentences, 300)
    sentence_matrix = np.vstack(sentence_vector)


    # we are choosing between these clusters for our k in k-means clustering
    range_n_clusters = [3, 4, 5, 6, 7]
    silhouette_scores = []

    for k in range_n_clusters:
        kmeans = KMeans(n_clusters=k, random_state=22)
        labels = kmeans.fit_predict(sentence_matrix)
        sil_avg = silhouette_score(sentence_matrix, labels)
        print(f"k = {k}, average silhouette score = {sil_avg:.3f}")
        silhouette_scores.append(sil_avg)

    plt.plot(range_n_clusters, silhouette_scores, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Average silhouette score")
    

    # Pick a k (e.g. the best silhouette score or just 3)
    k = 5
    kmeans = KMeans(n_clusters=k, random_state=22, n_init=10)

    #y axis of the graph for generating k means clusters
    y_kmeans = kmeans.fit_predict(sentence_matrix)

    #calculating sentence with minimum distance from centroid. This will be the representative sentence
    top_sentence_indexes = []
    #iterating over every sentence in the cluster and calculating distance between that sentence and the centroid
    for i in range(k):
        my_dict = {}
        for j in range(len(y_kmeans)):
            if y_kmeans[j] == i:
                # distance from sentence j to centroid i
                my_dict[j] = distance.euclidean(kmeans.cluster_centers_[i], sentence_matrix[j])
        #finds lowest distance
        min_index = min(my_dict, key=my_dict.get)
        #appends that sentence index to the list. This is the representativie sentence for the cluster
        top_sentence_indexes.append(min_index)
    summary = [article[index] for index in sorted(top_sentence_indexes)]
    print(". ".join(summary))
    #plt.show()



def TfIDF_summary(article):
    #this is a (# sentences, # vocabulary) sparse matrix (because most values are 0 so it doesn't include them to save memory)
    #each row is a sentence, and each column is the TFIDF score for every vocab word in said sentence
    transformed_article = vec_TfIDF.transform(article)

    #return a (1, m) matrix where it has one row and every column is the average of a single word across all sentences
    centroid = transformed_article.mean(axis = 0)
    #[:,0] takes all the rows, so we still have all the scores but then just takes the score
    # it flattens a 2d list with scores as a sublist into just a list of scores
    cos_similarity = cosine_similarity(transformed_article, np.array(centroid))[:, 0]

    best_indexes = np.argsort(-cos_similarity)[:3]
    best_indexes = sorted(best_indexes)
    summary = [article[index] for index in best_indexes]
    return summary

summary = ". ".join(TfIDF_summary(train_articles[3]))




        







    
        







