import tensorflow as tf
from tensorflow import keras

def BiLSTMCNN():
    model = tf.keras.models.load_model('models/bilstmcnn_fixtypo.h5')
    model_text = "bilstmcnn"

    return model, model_text


def BiLSTM():
    model = tf.keras.models.load_model('models/bilstm_fixtypo.h5')
    model_text = "bilstm"

    return model, model_text


def BiGRUCNN():
    model = tf.keras.models.load_model('models/bigrucnn_fixtypo.h5')
    model_text = "bigrucnn"

    return model, model_text