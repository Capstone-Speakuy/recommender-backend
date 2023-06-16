import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import convert_to_tensor, float32
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def get_mentor_recommendation(connection, user_id, user_text):
    mentor_df = get_mentors(connection)
    assigned_mentor_ids = get_assigned_mentor_ids(connection)
    mentor_df = mentor_df[~mentor_df['id'].isin(assigned_mentor_ids)]
    mentor_df['similiarity'] = 0
    mentor_df['similiarity'] = get_mentor_similiarity(user_text, mentor_df['description'].tolist())
    
    mentor_df.sort_values(by=['similiarity'], ascending=False, inplace=True)
    selected_mentor_df = mentor_df.iloc[:100].copy()

    consumed_df = selected_mentor_df.copy()
    consumed_df.rename(columns={'total_job':'work_total', 'salary_per_hour': 'salary'}, inplace=True)
    consumed_df['similiarity'] = consumed_df['similiarity'].apply(lambda x: np.array([x]))
    scaler = MinMaxScaler()
    consumed_df[["earned", "success_rate", "salary", "work_total", "total_hours"]] = scaler.fit_transform(consumed_df[["earned", "success_rate", "salary", "work_total", "total_hours"]])
    X_consume = consumed_df[["earned", "success_rate", "salary", "work_total", "total_hours", "similiarity"]]
    X_consume = convert_to_tensor(X_consume.values, dtype=float32)

    model = load_model(os.path.join(os.getcwd(), 'model', 'recommendation_model.h5'), compile=False)
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mse"])
    prediction = model.predict(X_consume)
    selected_mentor_df['predicted_avg_rate'] = prediction
    selected_mentor_df.sort_values(by=['predicted_avg_rate'], ascending=False, inplace=True)

    top_mentor_df = selected_mentor_df.iloc[:10].copy()
    top_mentor_df.drop(columns=['username', 'password', 'skills'], inplace=True)
    top_mentor_df['created_at'] = top_mentor_df['created_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
    top_mentor_df['updated_at'] = top_mentor_df['updated_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
    return top_mentor_df.to_dict(orient='records')

def get_mentors(connection):
    mentor_df = pd.read_sql("select * from mentors", connection)
    return mentor_df

def get_assigned_mentor_ids(connection):
    assigned = pd.read_sql("SELECT mentor_id FROM mentor_mentee", connection)
    return assigned['mentor_id'].tolist()

def get_mentor_similiarity(mentee_description, mentor_descriptions):
    vocab_size = 10000
    trunc_type='post'
    oov_tok = "<OOV>"
    sentences = [mentee_description] + mentor_descriptions

    tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(sentences)

    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, padding=trunc_type)

    similiarities = cosine_similarity(
    [padded[0]],
    padded[1:]
    )

    return similiarities[0]