import pandas as pd
import spacy
from multiprocess import Pool, cpu_count

# Load the spaCy model once to avoid repeated loading
nlp = spacy.load(r"C:\Users\User\Downloads\en_core_web_sm-3.8.0\en_core_web_sm-3.8.0\en_core_web_sm\en_core_web_sm-3.8.0")

# Preprocess and extract locations - same as before
def preprocess_and_extract_locations(text):
    doc = nlp(text.lower())

    # Extract tokens
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]

    # Extract geographical locations (GPE entities)
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]

    return ' '.join(tokens), locations

# Function to process rows in parallel
def process_row(row):
    return preprocess_and_extract_locations(row['Full_text'])

# Parallel processing function using multiprocess
def process_dataframe_parallel(df):
    # Use the maximum number of available CPU cores
    num_cores = cpu_count()
    print(f"Using {num_cores} cores for parallel processing...")

    # Convert DataFrame to dictionary for parallel processing
    rows = df.to_dict('records')

    # Use multiprocess Pool for parallel processing
    with Pool(num_cores) as pool:
        results = pool.map(process_row, rows)

    # Assign results back to the DataFrame
    df[['processed_text', 'locations']] = pd.DataFrame(results, index=df.index)

# Example DataFrame (replace this with your actual data)
if __name__ == "__main__":
    # Sample DataFrame (replace with your actual data)

    df = pd.read_csv(r"C:\Users\User\Downloads\processed_output.csv")


    # Run parallel processing
    process_dataframe_parallel(df)

    # Check the modified DataFrame
    df.to_csv(r"C:\Users\User\Downloads\tokens.csv",index=False)