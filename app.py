import streamlit as st
import joblib
import re

# Load the saved model and tools
model = joblib.load('genre_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
mlb = joblib.load('label_binarizer.pkl')

# Same cleaning function you used before
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_genre(text):
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])
    pred = model.predict(vec)
    return mlb.inverse_transform(pred)[0]

# --- Web page layout starts here ---
st.title("🎬 Movie Genre Predictor")
st.write("Enter a movie plot below and I'll guess the genre(s)!")

user_input = st.text_area("Movie plot:")

if st.button("Predict Genre"):
    if user_input.strip() == "":
        st.warning("Please enter a plot first.")
    else:
        genres = predict_genre(user_input)
        if len(genres) == 0:
            st.write("Couldn't confidently predict a genre.")
        else:
            st.success(f"Predicted Genre(s): {', '.join(genres)}")