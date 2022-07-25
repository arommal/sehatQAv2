import numpy as np
import pandas as pd
from numpy.linalg import norm
import time
from scipy.spatial import distance

from preprocessing import tokenize, embed_for_similarity
from prediction import get_id_predictions

def cosine_similarity(A, B):
    '''
    Calculate cosine similarity score of two texts
    '''
    cosine = np.dot(A,B)/(norm(A)*norm(B))
    return cosine


def get_similarity2(s1, s2, ft):
    '''
    Get cosine similarity score of two texts
    '''
    s1_emb = embed_for_similarity(s1, ft)
    s2_emb = embed_for_similarity(s2, ft)

    return cosine_similarity(s1_emb, s2_emb)


def get_candidates_as_df(predictions, labels_list):
    '''
    Konstruksi DataFrame dari kelompok data dengan label yang sama dengan hasil klasifikasi
    '''
    compare_df = pd.DataFrame(columns=['user', 'title', 'question', 'answer', 'topics', 'topics_num', 'labelmap', 'title_clean', 'question_clean', 'text_clean', 'answer_clean'])

    prediction_ids = get_id_predictions(predictions)
    for i in prediction_ids:
        class_df = pd.read_csv("data/" + labels_list[i] + ".csv")
        compare_df = pd.concat([compare_df, class_df], ignore_index=True)
            
    return compare_df


def retrieve_answers(question: str, compare_df: pd.DataFrame, ft):
    '''
    Mendapatkan rekomendasi jawaban dalam format dictionary
    '''
    SIM_THRESHOLD = 0.9
    curr_max_cos = SIM_THRESHOLD
    counts = 0

    result = {}
    result["sim_question"] = []
    result["answer"] = []
    result["similarity"] = []
    
    for index, row in compare_df.iterrows():
        cos_title = get_similarity2(question, str(row['title']), ft)
        cos_question = get_similarity2(question, str(row['question']), ft)
        cos_titleclean = get_similarity2(question, str(row['title_clean']), ft)
        cos_questionclean = get_similarity2(question, str(row['question_clean']), ft)
        cos_text = get_similarity2(question, str(row['text_clean']), ft)

        # Update curr_max_cos
        if cos_title > curr_max_cos or cos_question > curr_max_cos or cos_text > curr_max_cos or cos_titleclean > curr_max_cos or cos_questionclean > curr_max_cos:
            result["sim_question"].append(row['question'])
            result["answer"].append(row['answer_clean'])
            cos = max([cos_title, cos_question, cos_titleclean, cos_questionclean, cos_text])
            result["similarity"].append(cos)
            counts += 1
            curr_max_cos = cos

    if (counts == 0):
        default_answer = "Maaf, tidak terdapat jawaban yang sesuai"
        result["answer"].append(default_answer)
    
    return result
