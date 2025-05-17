# ReTweetify: Turkish Hate Speech Classification, Explanation & Rewriting 🇹🇷

This project leverages **Multilingual BERT**, **LoRA fine-tuning**, and **SHAP explainability** to **classify, interpret, and rewrite Turkish tweets** that contain hate speech. 
It combines modern NLP tools with transformers and parameter-efficient tuning to handle sensitive content in Turkish social media posts.

## 🔍 Project Description
- **Dataset**: 16k Turkish hate speech tweets classified into hateful (nefret) and not hateful (hiçbiri).
- **Model**: `bert-base-multilingual-cased` with PEFT/LoRA for efficient fine-tuning.
- **Explainability**: SHAP token-level explanations visualize what contributed to hate predictions.
- **Rewriting**: Hateful content is paraphrased using LLaMA-3 to preserve tone but remove offensive phrases.

---

## 📊 Dataset
Dataset is publicly available on **Kaggle**:  
🔗 [Turkish Hate Speech Tweets Dataset](https://www.kaggle.com/datasets/musadiqpashak/turkish-hatespeech-tweets)

---

## 🛠️ Features
- **EDA & Visualizations**: Tweet length, word count, label distributions, word cloud.
- **Training Setup**: 4-bit quantized BERT + LoRA (PEFT).
- **Metrics**: Accuracy, F1-macro, precision, recall.
- **Inference**: Classifies tweets into "nefret" or "hiçbiri".
- **Explanation**: SHAP values highlight which tokens drive classification.
- **Rewriting**: LLaMA-based generation of respectful alternative tweets.

---

## 📸 Screenshots

### 🔹 Model Results
![Results](https://github.com/MusadiqPasha/Turkish-Hate-Speech-Classification-Explanation/blob/main/output/results.png)

### 🔹 SHAP Explainability
![SHAP Explanability](https://github.com/MusadiqPasha/Turkish-Hate-Speech-Classification-Explanation/blob/main/output/shap.png)

### 🔹 Rewriting a Hate Tweet
![Re-writing Hate Tweet](https://github.com/MusadiqPasha/Turkish-Hate-Speech-Classification-Explanation/blob/main/output/rewrite.png)


---
## 🧪 Inference Sample
```python
text = "Bu kadar kötü bir şey olabilir mi?"
result = classify_text(text)
print(result)
