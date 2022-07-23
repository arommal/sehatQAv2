import pandas as pd
import time
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from preprocessing import load_tokenizer, load_fasttext_vocab, clean_text
from prediction import get_label_predictions, get_demap_predictions
from retrieval import  get_candidates_as_df, retrieve_answers
from classification_model import BiLSTMCNN


app = FastAPI(
    title="SehatQA",
    description="Simple API to demonstrate predicting medical question-answer",
    version="0.1",
)
app.mount("/public", StaticFiles(directory="public"), name="public")

templates = Jinja2Templates(directory="public/")

@app.get("/")
def index(request: Request):
    '''
    Main function to read user input
    '''
    return templates.TemplateResponse("index.html", context={'request': request})


@app.post("/result")
def result(request: Request, question: str = Form(...)):
    '''
    Function to process input from user
    '''
    result = predict(question)
    return templates.TemplateResponse("result.html", {'request': request, 'result': result})


@app.get("/predict")
def predict(question: str):
    '''
    Fungsi utama untuk melakukan prediksi hingga ekstraksi rekomendasi jawaban
    '''
    # Definisi parameter utama untuk klasifikasi
    CLASSIFICATION_MODEL = BiLSTMCNN()
    TOKENIZER = load_tokenizer()
    FT = load_fasttext_vocab()

    # Menyimpan identifier 10 topik utama
    df_labels = pd.read_csv('data/labels.csv')
    labels_list = df_labels['label'].values
    
    start_time = time.time()

    # '''Baseline: No classification process'''
    # compare_df = pd.read_csv('data/alodokter_final_selected_7.csv')

    # Clean question
    cleaned_question = clean_text(question)

    '''Proposed: With classification process'''
    # Mendapatkan hasil klasifikasi (predictions: 0's and 1's, predictions_demap: array string)
    predictions = get_label_predictions(CLASSIFICATION_MODEL, TOKENIZER, cleaned_question)
    predictions_demap = get_demap_predictions(predictions[0], labels_list)
    
    # Mendapatkan kandidat pasangan pertanyaan-jawaban
    compare_df = get_candidates_as_df(predictions[0], labels_list)

    # Memilih rekomendasi jawaban
    result = retrieve_answers(cleaned_question, compare_df, FT)

    # Menambahkan elemen ke hasil dalam JSON
    result["question"] = question
    result["labels"] = predictions_demap

    end_time = time.time()
    retrieval_duration = end_time - start_time
    result["time"] = retrieval_duration

    return result