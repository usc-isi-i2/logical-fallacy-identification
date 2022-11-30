import pandas as pd

path = "/cluster/raid/home/darshan.deshpande/curriculum_learning_training/evaluate_sentence-transformers-nli-distilbert-base.csv"
df = pd.read_csv(path)

fine_classes = ['fallacy of red herring', 'faulty generalization', 'ad hominem', 'false causality', 'circular reasoning', 'ad populum', 'fallacy of credibility', 'appeal to emotion', 'fallacy of logic', 'intentional', 'fallacy of extension', 'false dilemma', 'equivocation', 'prejudicial language', 'slothful induction', 'fallacy of slippery slope']

def mapper(val):
    return fine_classes[val]

df["ground_truth"] = df["ground_truth"].apply(mapper)
df["prediction"] = df["prediction"].apply(mapper)
print(df.to_csv(path))