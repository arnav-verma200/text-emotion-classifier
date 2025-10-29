---

# 🧠 Emotion Detection with Machine Learning

A text-based emotion classification model built using **Support Vector Machine (SVM)** and **TF-IDF** vectorization. This project classifies human emotions into six categories:

> **sadness | joy | love | anger | fear | surprise**

---

## 📌 Features

✅ Preprocessed dataset using TF-IDF
✅ Trained SVM (Linear Kernel) classifier
✅ Model evaluation using multiple visualizations:

* Confusion Matrix (Counts & Normalized)
* ROC Curves (One-vs-Rest)
* Precision-Recall Curves
* Classification report
  ✅ Dark-theme graphical dashboard
  ✅ Saved model for future predictions
  ✅ CLI tool for real-time emotion prediction

---

## 📂 Project Structure

```
📦 Emotion-Classifier-SVM
 ┣ 📄 svm_train.py               # Training + Visualization
 ┣ 📄 predict_emotion.py         # Load model & predict emotions
 ┣ 📄 svm_emotion_model.pkl      # Saved trained model
 ┣ 📄 tfidf_vectorizer.pkl       # Saved TF-IDF vectorizer
 ┣ 🖼️ svm_metrics_visualization.png  # Combined graph output
 ┗ 📄 README.md
```

---

## 🛠️ Installation

### ✅ 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/emotion-svm.git
cd emotion-svm
```

### ✅ 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Train the Model (Optional)

Run the file if you want to retrain and regenerate the visualizations:

```bash
python svm_train.py
```

This will generate:

✅ Trained model `.pkl` files
✅ Evaluation report plots (single combined image)

---

## 🔍 Use the Model for Predictions

After training, test new sentences using:

```bash
python predict_emotion.py
```

Example:

```
Enter a sentence: I am feeling amazing today!
Predicted Emotion: JOY
```

---

## 📊 Visualization Example

The project generates a single dashboard image with all evaluation metrics:

> 🖼️ `svm_metrics_visualization.png`
> (inferno heatmap, ROC, PR curve, accuracy plot, report block)

---

## 📚 Dataset

This project uses the **Emotion dataset** from 🤗 Hugging Face:

```
dair-ai/emotion
```

Labels:

| ID | Emotion  |
| -- | -------- |
| 0  | sadness  |
| 1  | joy      |
| 2  | love     |
| 3  | anger    |
| 4  | fear     |
| 5  | surprise |

---

## ✅ Technology Stack

| Component          | Library                |
| ------------------ | ---------------------- |
| Model              | SVM (Linear Kernel)    |
| Feature Extraction | TF-IDF                 |
| Dataset            | HuggingFace `datasets` |
| Visuals            | Matplotlib, Seaborn    |
| Runtime            | Python 3.8+            |

---

## 🏁 Future Improvements

* ✅ Deploy as a REST API (Flask / FastAPI)
* ✅ Convert into a chat-based UI
* ✅ Add more advanced models (BERT, RoBERTa)
* ✅ Streamlit web app for easy interaction

---

## 🤝 Contributions

Contributions are welcome!
Feel free to submit issues or PRs to improve the project.

---

## 📜 License

This project is licensed under the **MIT License** — free to use & modify.

---

### ⭐ If you like this project — give it a star on GitHub!

---
Would you like a **Light theme version** of the graphs or a **Streamlit UI** next? 🚀
