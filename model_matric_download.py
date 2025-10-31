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
from sklearn.model_selection import GridSearchCV


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
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=20000)
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

#testing and seeing stuff
print(X_test[1])

#training model type shit (using balanced class weights + tuning)
params = {'C': [0.1, 1, 10]}
grid = GridSearchCV(SVC(kernel="linear", probability=True, class_weight="balanced"), param_grid=params, cv=3, n_jobs=-1, verbose=1)
grid.fit(X_train, train_labels)
svm_model = grid.best_estimator_
print(f"Best SVM Params: {grid.best_params_}")

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

#Confusion Matrix (Counts)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="inferno", xticklabels=emotion_names, yticklabels=emotion_names)
plt.title("Confusion Matrix (Counts)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_counts.png", dpi=300)
plt.close()

#Normalized Confusion Matrix
plt.figure(figsize=(10,8))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="viridis", xticklabels=emotion_names, yticklabels=emotion_names)
plt.title("Confusion Matrix (Normalized %)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_normalized.png", dpi=300)
plt.close()

#Probability matrix for ROC + PR
y_prob = svm_model.predict_proba(X_test)
y_test_bin = label_binarize(test_labels, classes=range(len(emotion_names)))

#ROC Curves (Multiclass)
plt.figure(figsize=(10,8))
for i in range(len(emotion_names)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    plt.plot(fpr, tpr, label=f"{emotion_names[i]} (AUC= {auc(fpr,tpr):.2f})")

plt.plot([0,1],[0,1],'--')
plt.title("ROC Curves (One-vs-Rest)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=300)
plt.close()

#Precision-Recall Curves
plt.figure(figsize=(10,8))
for i in range(len(emotion_names)):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_prob[:, i])
    plt.plot(recall, precision, label=emotion_names[i])

plt.title("Precision-Recall Curves")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.tight_layout()
plt.savefig("precision_recall_curves.png", dpi=300)
plt.close()

#Accuracy Bar Plot
plt.figure(figsize=(6,6))
plt.bar(["Accuracy"], [acc], color="cyan")
plt.ylim(0,1)
plt.title(f"Overall Accuracy: {acc:.2%}")
plt.ylabel("Score")
plt.tight_layout()
plt.savefig("accuracy_plot.png", dpi=300)
plt.close()

#Classification Score Summary Text
report = classification_report(test_labels, svm_pred, target_names=emotion_names)
plt.figure(figsize=(8,6))
plt.axis("off")
plt.text(0, 0.5, report, fontsize=12)
plt.title("Classification Report")
plt.tight_layout()
plt.savefig("classification_report.png", dpi=300)
plt.close()


#downloading the model and stuff
import joblib

joblib.dump(svm_model, "svm_emotion_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model and Vectorizer Saved Successfully!")
