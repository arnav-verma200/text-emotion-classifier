from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_recall_curve
import seaborn as sns
import numpy as np
from sklearn.preprocessing import label_binarize


dataset = load_dataset("dair-ai/emotion")

#split into train test
train_data = dataset["train"]
test_data = dataset["test"]

#tried printing stuff
print(train_data[0])


#tain test ts
train_texts = [x['text'] for x in train_data]
train_labels = [x['label'] for x in train_data]
test_texts = [x['text'] for x in test_data]
test_labels = [x['label'] for x in test_data]

#try printing stuff to see
print(train_texts[3])
print(train_labels[3])


#emotions
emotion_names = dataset["train"].features["label"].names
print(emotion_names)

#converting the txt into numbers type shit so machine can understand stuff
vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

#testing and seeing stuff
print(X_test[1])

#training model type shit
svm_model = SVC(kernel="linear", probability=True)
svm_model.fit(X_train, train_labels)

#prediction
svm_pred = svm_model.predict(X_test)
print("SVM Classification Report:")
print(classification_report(test_labels, svm_pred, target_names=emotion_names))

#confusion matrix
cm = confusion_matrix(test_labels, svm_pred)
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

acc = accuracy_score(test_labels, svm_pred)



# Dark Theme
plt.style.use("dark_background")
fig, axes = plt.subplots(3, 2, figsize=(24, 26))
fig.suptitle("SVM Emotion Classifier - Evaluation Metrics (Dark Theme)", fontsize=24)



#Confusion Matrix Counts
sns.heatmap(cm, annot=True, fmt="d", cmap="inferno", ax=axes[0,0],
            xticklabels=emotion_names, yticklabels=emotion_names)
axes[0,0].set_title("Confusion Matrix (Counts)")
axes[0,0].set_xlabel("Predicted")
axes[0,0].set_ylabel("Actual")

#Normalized Confusion Matrix
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="viridis", ax=axes[0,1],
            xticklabels=emotion_names, yticklabels=emotion_names)
axes[0,1].set_title("Confusion Matrix (Normalized %)")
axes[0,1].set_xlabel("Predicted")
axes[0,1].set_ylabel("Actual")

#Probability matrix for ROC + PR
y_prob = svm_model.predict_proba(X_test)
y_test_bin = label_binarize(test_labels, classes=range(len(emotion_names)))

#ROC Curves (Multiclass)
for i in range(len(emotion_names)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    axes[1,0].plot(fpr, tpr, label=f"{emotion_names[i]} (AUC= {auc(fpr,tpr):.2f})")

axes[1,0].plot([0,1],[0,1],'--')
axes[1,0].set_title("ROC Curves (One-vs-Rest)")
axes[1,0].set_xlabel("False Positive Rate")
axes[1,0].set_ylabel("True Positive Rate")
axes[1,0].legend()

#Precision-Recall Curves
for i in range(len(emotion_names)):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_prob[:, i])
    axes[1,1].plot(recall, precision, label=emotion_names[i])

axes[1,1].set_title("Precision-Recall Curves")
axes[1,1].set_xlabel("Recall")
axes[1,1].set_ylabel("Precision")
axes[1,1].legend()

#Accuracy Bar Plot
axes[2,0].bar(["Accuracy"], [acc], color="cyan")
axes[2,0].set_ylim(0,1)
axes[2,0].set_title(f"Overall Accuracy: {acc:.2%}")
axes[2,0].set_ylabel("Score")

#Classification Score Summary Text
report = classification_report(test_labels, svm_pred, target_names=emotion_names)
axes[2,1].axis("off")
axes[2,1].text(0, 0.5, report, fontsize=12)

plt.tight_layout()
plt.subplots_adjust(top=0.93)

#downloading the model and stuff
import joblib

joblib.dump(svm_model, "svm_emotion_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model and Vectorizer Saved Successfully!")