from preprocessing import tokenize

def get_label_predictions(model, tokenizer, cleaned_question):
    '''
    Mendapatkan array 0/1 sebagai hasil prediksi klasifikasi
    '''
    tokenized_question = tokenize(cleaned_question, tokenizer)
    if model[1] == "bilstmcnn" or model[1] == "bigrucnn":
        predictions = model[0].predict([tokenized_question, tokenized_question])
    elif model[1] == "bilstm" or model[1] == "bigru":
        predictions = model[0].predict(tokenized_question)

    output = (predictions > 0.5).astype(int)
    
    return output


def get_id_predictions(predictions: list):
    '''
    Get indexed version of label prediction
    '''
    ids = []
    for index, i in enumerate(predictions):
        if i == 1:
            ids.append(index)

    return ids


def get_demap_predictions(pred: list, labels_list: list):
    '''
    Get text version of label prediction
    '''
    idx = 0
    labels = []
    for i in pred:
        if i == 1:
            labels.append(labels_list[idx])
        idx = idx+1
        
    return labels