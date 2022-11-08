import pandas as pd
import numpy as np
from tqdm import tqdm
import nlpaug.augmenter.word as naw
import torch

def augment(samples, class_, df, aug, times):
    counter = 0
    np.random.shuffle(samples)
    repetitions = (times // len(samples)) + 1

    for s in tqdm(np.repeat(samples, repetitions)):
        if counter == times:
            break
        df = df.append({"text": aug.augment(s), "label":class_}, ignore_index=True)
        counter += 1
  
    return df

def main(train_csv, output_csv, text_column="text", label_column="label"):

    df_train = pd.read_csv(train_csv, usecols=[text_column, label_column])

    value_counts = df_train[label_column].value_counts().to_dict()
    print(f"Before augmentation: {value_counts}")

    max_val = max(value_counts.values())

    # Decides how much to augment for every class. Change the `max_val` to change this
    differences = {key: max_val-value for key,value in value_counts.items()}

    # Check if GPU is available
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Initialize augmenter
    aug = naw.ContextualWordEmbsAug(model_path='roberta-base', top_k=20, aug_p=0.2, action="substitute", device=device)

    # Augment classes
    for key, value in differences.items():
        print(f"Augmenting class: {key}")
        df_train = augment(df_train[df_train[label_column]==key][text_column].to_numpy(), key, df_train, aug, value)

    # Augmented dataset count
    print(f"After augmentation: {df_train[label_column].value_counts().to_dict()}")

    df_train.to_csv(output_csv, index=False)


if __name__ =="__main__":
    input_csv = "ptc_slc_without_none_with_context/coarse/train.csv"
    output_csv = "ptc_slc_without_none_with_context_nlpaug/coarse/train.csv"
    main(input_csv, output_csv)

    input_csv = "ptc_slc_without_none_with_context/fine/train.csv"
    output_csv = "ptc_slc_without_none_with_context_nlpaug/fine/train.csv"
    main(input_csv, output_csv)
    