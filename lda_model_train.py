import os
import pickle
from gensim import corpora
from gensim.models.ldamulticore import LdaMulticore

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

# ---------------------
# Train LDA model


# ---------------------
def train():
    num_topics = 15
    lda_model = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=10,
        random_state=42,
        workers=11
    )
    print("LDA model trained successfully.")
    lda_model.save(model_save_path)

# ---------------------
# Save the trained model
# ---------------------
if __name__=='__main__':
    train()
    print(f"LDA model saved at: {model_save_path}")
