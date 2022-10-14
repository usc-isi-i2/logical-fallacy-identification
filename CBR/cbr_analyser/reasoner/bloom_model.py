from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom")

model = AutoModelForSequenceClassification.from_pretrained(
    "bigscience/bloom")


print('loaded')
