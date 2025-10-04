# Toxic Comment Classification  

This project builds and compares two machine learning models to classify whether online comments are **toxic** or **not toxic**. The dataset contains ~4000 comments from Reddit, Twitter/X, and YouTube, each annotated by multiple annotators.  

We implemented two models:  
1. **Baseline:** TF-IDF + Logistic Regression  
2. **Advanced:** Fine-tuned BERT (bert-base-uncased)  

The final submission requires predictions for a hidden test dataset in CSV format (`platform_id, prediction`).  

---

## 🚀 Project Structure  

- `z639_assignment1_training.json` → Training dataset (labeled)  
- `z639_assignment1_test.json` → Test dataset (unlabeled, only `platform_id` and `text`)  
- `notebooks/` → Jupyter/Colab notebooks for data processing and training  
- `submission.csv` → Final predictions for test set  

---

## 📊 Methods  

We started with **data processing**, which included cleaning text (removing URLs and extra whitespace) and aggregating multiple annotator labels into a single binary label using majority vote. The dataset was split into train, validation, and test subsets in a stratified manner to preserve class balance.  

For the **baseline**, we trained a TF-IDF + Logistic Regression model using unigram and bigram features (max 20,000). Logistic regression was trained with class balancing to account for label imbalance.  

For the **advanced model**, we fine-tuned a BERT transformer (`bert-base-uncased`). Comments were tokenized with a maximum length of 128 tokens. Training ran for three epochs with the AdamW optimizer and a learning rate of 2e-5. Early stopping was applied to prevent overfitting, with the best model selected at epoch 1.  

---

## 📈 Results  

| Model                        | Accuracy | Precision (toxic) | Recall (toxic) | F1 (toxic) |
|-------------------------------|----------|-------------------|----------------|------------|
| TF-IDF + Logistic Regression | ~0.74    | ~0.49             | ~0.52          | ~0.50      |
| Fine-tuned BERT (best epoch) | ~0.79    | ~0.60             | ~0.62          | ~0.61      |

BERT significantly outperformed the TF-IDF baseline, particularly in detecting toxic comments, where F1 improved by over 10 points.  

---

## 🛠️ How to Run  

1. Clone the repo and install dependencies:  
   ```bash
   git clone https://github.com/freny24/toxic-comment-classification.git
   cd toxic-comment-classification
   pip install -r requirements.txt

2. Run the training notebooks:
notebooks/baseline_tfidf.ipynb → trains TF-IDF + Logistic Regression
notebooks/bert_finetune.ipynb → fine-tunes BERT

Generate predictions on the test set and export as CSV:
python generate_submission.py


The output will be a file named submission.csv with the format:

platform_id,prediction
12345,true
12346,false
12347,true
...



## 📌 Key Takeaways

TF-IDF + Logistic Regression provides a simple but limited baseline.

Fine-tuned BERT captures contextual meaning better, leading to higher recall and F1.

Overfitting occurs quickly on small datasets; early stopping is crucial.
