import os
import pickle
from gensim import corpora
from gensim.models.ldamulticore import LdaMulticore
import pandas as pd
import multiprocessing as mp
from gensim.models import LdaModel
import re
# ---------------------
# Define file paths
# ---------------------
dictionary_path = r"C:\Users\User\Downloads\news_dictionary_final.dict"  # Update your path
corpus_path = r"C:\Users\User\Downloads\news_corpus_finall.mm"  # Update your path
model_save_path = r"C:\Users\User\Downloads\lda_model.model"  # Update path

# ---------------------
# Load dictionary
# ---------------------
dictionary = corpora.Dictionary.load(dictionary_path)
print("Dictionary loaded successfully.")

# ---------------------
# Load corpus (.mm format)
# ---------------------
corpus = corpora.MmCorpus(corpus_path)
print("Corpus loaded successfully.")


def string_to_token_list(input_string):
    """
    Converts a string to a list of tokens, removing extra quotes, punctuation, and whitespace.

    Args:
        input_string: The input string.

    Returns:
        A list of cleaned tokens.
    """
    # Split string by whitespace and clean tokens
    tokens = input_string.split()
    cleaned_tokens = [
        re.sub(r"[^a-zA-Z0-9]", "", token.strip().lower())
        for token in tokens
        if token.strip()
    ]
    return [token for token in cleaned_tokens if token]
# Function to get the most probable topic for each document
def get_dominant_topic(text, dictionary, lda_model):
    bow = dictionary.doc2bow(text)  # Convert text into BoW representation
    topics = lda_model.get_document_topics(bow)  # Get topic distribution

    if topics:
        return max(topics, key=lambda x: x[1])[0]  # Get topic ID with highest probability
    return -1  # Return -1 if no topics found


# Function to process the dataset in parallel
def parallel_topic_assignment(texts, dictionary, lda_model, num_workers=None):
    if num_workers is None:
        num_workers = mp.cpu_count()  # Use all available CPU cores

    with mp.Pool(processes=num_workers) as pool:
        labels = pool.starmap(get_dominant_topic, [(text, dictionary, lda_model) for text in texts])

    return labels


if __name__ =="__main__":
    df=pd.read_csv(r"C:\Users\User\Downloads\tokens.csv")
    df=df[:5000]
    df['token'] = df['processed_text'].apply(string_to_token_list)
    lda_model = LdaModel.load(r"C:\Users\User\Downloads\lda_model.model")


    df['label'] = parallel_topic_assignment(df['token'], dictionary, lda_model)
    df.to_csv(r"C:\Users\User\Downloads\training.csv",index=False)
