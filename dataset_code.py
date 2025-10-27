from datasets import load_dataset

dataset = load_dataset("dair-ai/emotion")

# Split
train_data = dataset["train"]
test_data = dataset["test"]

# Example sample
print(train_data[0])
