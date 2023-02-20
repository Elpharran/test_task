import json
import urllib

import faiss
import numpy as np
import pandas as pd
import requests
import streamlit as st
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


st.set_page_config(
    page_title="Simple news search",
    page_icon="🔍",
    layout="wide"
)

descr, meme = st.columns([10, 2])

with descr:
    st.title("Exploring Russian news from the Gazeta website 👀 ")
    st.header("")

    with st.expander("ℹ️ - About this app", expanded=True):

        st.write(
            """     
    - This app is an easy-to-use interface built in Streamlit for testing simple reverse text search through the summaries of the news articles from **Gazeta.ru**.
    - It relies on a pre-trained neural network encoder [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)
    and Facebook AI Similarity Search [FAISS](https://github.com/facebookresearch/faiss) to efficiently search for top 10 similar descriptions in decreasing cosine similarity.
    - The [data](https://huggingface.co/datasets/IlyaGusev/gazeta) was collected by Ilya Gusev in 2020. All rights belong to www.gazeta.ru.
            """
        )

        st.markdown("")

with meme:
    st.image("meme.jpg")


@st.cache_resource
def load_encoder():
    return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")


@st.cache_data
def load_data():
    dataset = load_dataset("IlyaGusev/gazeta", revision="v1.0")
    data = pd.DataFrame(dataset["train"])

    folder_url = 'https://disk.yandex.ru/d/spPvdtAvDVbpIQ'
    file_url = 'https://disk.yandex.ru/d/7Wk5PLWX8jEsYw'
    url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download' + '?public_key=' + urllib.parse.quote(folder_url) + '&path=/' + urllib.parse.quote(file_url)

    r = requests.get(url)
    h = json.loads(r.text)['href']
    index = faiss.read_index(h)
    
    return data, index


def get_top10(query, data, encoder, index):
    query_vector = encoder.encode([query])
    faiss.normalize_L2(query_vector)
    _, indices = index.search(query_vector, 10)

    for i, idx in enumerate(indices[0]):
        news = data["summary"].iloc[idx]
        url = data["url"].iloc[idx]
        title = data["title"].iloc[idx]
        date = data["date"].iloc[idx]
        lines = [f"**{i + 1}. {title},** \narticle from {date[:11]}, link {url}\n\n{news}\n"]
        st.write("\n".join(lines))


data, index = load_data()
encoder = load_encoder()

text_input = st.text_input("Let's find something :nerd_face:", "Британские учёные обнаружили новый вид рыб в Тихом океане")

if st.button("Submit"):
    get_top10(text_input, data, encoder, index)