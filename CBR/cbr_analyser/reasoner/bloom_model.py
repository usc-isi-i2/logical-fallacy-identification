from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom")

model = AutoModelForSequenceClassification.from_pretrained(
    "bigscience/bloom")


print('loaded')