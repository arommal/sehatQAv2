# 🤳🏽 SehatQA
This repository serves to document my undergraduate thesis project.

SehatQA is a web-based answer recommendation system. It uses and is trained on Alodokter question-answer data from 2014-2020. The system performs **three tasks** for every question input by a user:

##### 1. Topic classification
This task classifies user input into one or more topics (multi-label classification).
For development efficency and data limitation reasons, user question has to be under the 10 topics specified in [labels.csv](data/labels.csv). Classification is performed using a BiLSTM-CNN model.
🧬 BiLSTM-CNN training specifications:
- Model plot
- Batch size
- Epochs
- Train/Validation/Test data ratio

📈 BiLSTM-CNN performance:
- Precision
- Recall
- Macro F1
- Subset Accuracy
- Hamming Loss

##### 2. Similar questions selection
This task selects top 10 most similar questions.
Similarity between input question and each dataset question is evaluated using Cosine Similarity. Text is represented as vectors using pretrained embeddings from Word2Vec.

##### 3. Answer recommendation extraction
The answers from each selected similar questions are summarized extractively and presented back to the user as recommended answers.

### Requirements
1. Python 3.8.10, download [here](https://www.python.org/downloads/release/python-3810/)
2. Install packages from requirements.txt
```
pip install -r requirements.txt
```
3. Upgrade scikit-learn to v1.1.1
```
pip install scikit-learn==1.1.1
```
4. Install FastAPI, Uvicorn, Jinja2, and python-multipart to run system locally
```
pip install fastapi uvicorn jinja2 python-multipart
```