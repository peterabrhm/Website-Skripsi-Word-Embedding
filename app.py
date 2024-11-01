from flask import Flask, render_template, request, session
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import re
from tqdm import tqdm
import time
from threading import Lock

app = Flask(__name__)
app.secret_key = 'hai peter'

cancel_lock = Lock()
cancellation_flags = {}

def load_resources(word2vec_model_path, lookup_file_path, paragraph_lookup_path, vectorizer_path):
    # Load the Word2Vec model
    word2vec_model = KeyedVectors.load(word2vec_model_path, mmap='r')  
    
    # Load Metadata Lookup Table
    lookup_df = pd.read_csv(lookup_file_path)
    
    lookup_dict = {}
    for index, row in lookup_df.iterrows():
        author_id = row['id_author']
        author_name = row['author_name']
        book_id = row['id_book']
        book_name = row['book_name']
        chapter_id = row['id_chapter']
        chapter_name = row['chapter_name']

        if author_id not in lookup_dict:
            lookup_dict[author_id] = {'author_name': author_name, 'books': {}}
        if book_id not in lookup_dict[author_id]['books']:
            lookup_dict[author_id]['books'][book_id] = {'book_name': book_name, 'chapters': {}}
        lookup_dict[author_id]['books'][book_id]['chapters'][chapter_id] = chapter_name
    
    # Load Paragraph Lookup Table
    paragraph_df = pd.read_csv(paragraph_lookup_path)
    
    paragraph_dict = pd.Series(paragraph_df['Paragraph Content'].values, index=paragraph_df['Paragraph ID']).to_dict()
    
    # Load Vectorizer
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

    return word2vec_model, lookup_dict, paragraph_dict, vectorizer

# Load vectors from a txt file
def load_txt(folder_path):
    document_dict = {}  # Initialize the dictionary to store document vectors
    error_log = []
    
    # Collect all filenames first to know the total number of files for tqdm
    txt_files = []
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith('.txt'):
                txt_files.append(os.path.join(dirpath, filename))

    # Iterate through the text files with a progress bar
    for file_path in tqdm(txt_files, desc="Loading vectors from txt files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
                # Match paragraph sections with the format `=== Start of Paragraph X === {vector}`
                pattern = r'=== Start of Paragraph (\d+_\d+_\d+_\d+) ===\n\{([^\}]*)\}'
                matches = re.findall(pattern, content)
                
                for match in matches:
                    author_book_chapter_paragraph_id = match[0]  # The paragraph ID
                    vector_data = match[1].strip()  # The vector content

                    # Initialize lists to store the sparse and embedding vectors
                    sparse_vector = {}
                    embedding_vector = []

                    # Split the vector data by comma and colon to get index and tf-idf value pairs
                    vector_pairs = vector_data.split(',')
                    
                    for pair in vector_pairs:
                        pair = pair.strip()
                        try:
                            if ':' in pair:  # Sparse data (index:value pairs)
                                index, value = map(float, pair.split(':'))
                                sparse_vector[int(index)] = value
                            else:  # Embedding data (300-dimensional float values)
                                embedding_value = float(pair)
                                embedding_vector.append(embedding_value)
                        except Exception as e:
                            error_log.append(f"Error: {e} on {author_book_chapter_paragraph_id}")
                    
                    embedding_vector = np.array(embedding_vector)

                    document_dict[author_book_chapter_paragraph_id] = {'sparse': sparse_vector, 'embedding': embedding_vector}
                    
        except Exception as e:
            error_log.append(f"Failed to read {file_path}: {e}")
    
    return document_dict
    

def search(query, vectorizer, expanded_document_dict, lookup_dict, word2vec_model, paragraph_dict, top_n):
    # Transform query
    query_vector = vectorizer.transform([query]).toarray()[0]
    sparse_part = {i: val for i, val in enumerate(query_vector) if val > 0}

    query_tokens = query.split()
    embedding_part = np.mean([word2vec_model[word] for word in query_tokens if word in word2vec_model], axis=0)

    query_vector = {'sparse': sparse_part, 'embedding': embedding_part}

    session_id = session.get('session_id')
    
    document_dict_similarities = {}
    # Compute similarities between the query and each document
    for idx, (doc_id, doc_vector) in enumerate(expanded_document_dict.items()):
        if idx % 1000 == 0:
            with cancel_lock:
                if cancellation_flags.get(session_id):
                    print("Search cancelled by the user.")
                    return []
            
        similarity = custom_cosine_similarity(query_vector, doc_vector)
        document_dict_similarities[doc_id] = similarity
    
    # Sort by similarity
    sorted_similarities = sorted(document_dict_similarities.items(), key=lambda item: item[1], reverse=True)
    
    # Store the top N results
    results = []
    for author_book_chapter_id, similarity in sorted_similarities[:top_n]:
        author_id, book_id, chapter_id, paragraph_id = author_book_chapter_id.split("_")
        author_name = lookup_dict[int(author_id)]['author_name']
        book_name = lookup_dict[int(author_id)]['books'][int(book_id)]['book_name']
        chapter_name = lookup_dict[int(author_id)]['books'][int(book_id)]['chapters'][int(chapter_id)]
        
        paragraph_content = paragraph_dict.get(author_book_chapter_id, "Paragraph not found")
        
        results.append({
            'similarity': round(similarity, 4),
            'author_name': author_name,
            'book_name': book_name,
            'chapter_name': chapter_name,
            'paragraph_content': paragraph_content
        })
    
    return results

def custom_cosine_similarity(query, doc):
    # Cosine similarity for sparse part
    query_sparse = query['sparse']
    doc_sparse = doc['sparse']

    dot_product_sparse = sum(query_sparse.get(k, 0) * doc_sparse.get(k, 0) for k in set(query_sparse) | set(doc_sparse))
    magnitude_query_sparse = np.sqrt(sum(v**2 for v in query_sparse.values()))
    magnitude_doc_sparse = np.sqrt(sum(v**2 for v in doc_sparse.values()))

    # Cosine similarity for embedding part
    # query_embedding = np.array(query['embedding'])
    # doc_embedding = np.array(doc['embedding'])
    
    query_embedding = query['embedding']
    doc_embedding = doc['embedding']
    
    dot_product_embedding = np.dot(query_embedding, doc_embedding)
    magnitude_query_embedding = np.linalg.norm(query_embedding)
    magnitude_doc_embedding = np.linalg.norm(doc_embedding)

    # Combine results
    total_dot_product = dot_product_sparse + dot_product_embedding
    total_magnitude_query = np.sqrt(magnitude_query_sparse**2 + magnitude_query_embedding**2)
    total_magnitude_doc = np.sqrt(magnitude_doc_sparse**2 + magnitude_doc_embedding**2)

    if total_magnitude_query == 0 or total_magnitude_doc == 0:
        return 0.0

    res = total_dot_product / (total_magnitude_query * total_magnitude_doc)
    
    return res

# Load resources and vectorizer
word2vec_model_path = '/Users/peterabraham/Downloads/Skripsi/Website - Word Embedding/data/word2vec-google-news-300-local.kv'
lookup_file_path = '/Users/peterabraham/Downloads/Skripsi/Website - Word Embedding/data/lookup_table.csv'
vectorizer_path = '/Users/peterabraham/Downloads/Skripsi/Website - Word Embedding/data/vectorizer-400.pkl'
vectors_path = '/Users/peterabraham/Downloads/Skripsi/Website - Word Embedding/data/full_data_vectors_txt'
lookup_paragraph_path = '/Users/peterabraham/Downloads/Skripsi/Website - Word Embedding/data/paragraph_lookup_table.csv'

word2vec_model, lookup_dict, paragraph_dict, vectorizer = load_resources(word2vec_model_path, lookup_file_path, lookup_paragraph_path, vectorizer_path)
document_dict = load_txt(vectors_path)

top_n = 10

@app.route('/cancel-search', methods=['POST'])
def cancel_search():
    session_id = session.get('session_id')
    with cancel_lock:
        cancellation_flags[session_id] = True
    return '', 204

@app.before_request
def assign_session_id():
    if 'session_id' not in session:
        session['session_id'] = str(time.time()) + str(np.random.rand())

@app.route('/', methods=['GET', 'POST'])
def index():
    session_id = session.get('session_id')
    with cancel_lock:
        cancellation_flags[session_id] = False

    if request.method == 'POST':
        query = request.form.get('query')
        if query:
            start_time = time.time()
            results = search(query, vectorizer, document_dict, lookup_dict, word2vec_model, paragraph_dict, top_n)
            search_time = round(time.time() - start_time, 4)
            return render_template('results.html', query=query, results=results, search_time=search_time)
        else:
            error = "Please enter a search query."
            return render_template('index.html', error=error)
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)