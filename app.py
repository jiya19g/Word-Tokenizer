import streamlit as st
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Streamlit app
def main():
    """Simple NLP App"""
    st.title("News Text NLP Processor")

    # Sidebar
    st.sidebar.subheader("About")
    st.sidebar.info("A simple NLP app for tokenization and visualization.")

    # NLP Processing Section
    st.info("Natural Language Processing of Text")
    raw_text = st.text_area("Enter News Text", "Type Here")

    nlp_task = ["Tokenization"]
    task_choice = st.selectbox("Choose NLP Task", nlp_task)

    if st.button("Analyze"):
        st.info("Original Text:")
        st.write(raw_text)

        docx = nlp(raw_text)

        if task_choice == 'Tokenization':
            result = [token.text for token in docx]
            st.json(result)

    if st.button("Tabulize"):
        docx = nlp(raw_text)
        tokens = [token.text for token in docx]
        lemma = [token.lemma_ for token in docx]
        pos = [token.pos_ for token in docx]

        new_df = pd.DataFrame(zip(tokens, lemma, pos), columns=['Token', 'Lemma', 'POS'])
        st.dataframe(new_df)

    if st.checkbox("WordCloud"):
        wordcloud = WordCloud(stopwords=set(STOPWORDS), background_color="white").generate(raw_text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot()

if __name__ == '__main__':
    main()
