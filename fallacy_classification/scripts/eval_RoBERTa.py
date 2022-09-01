# Importing the libraries needed
import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
import numpy as np
from tqdm import tqdm 
from sklearn.metrics import classification_report,  confusion_matrix
from torch import cuda
import argparse 

device = 'cuda' if cuda.is_available() else 'cpu'


class LogicalFallacy(Dataset):
    """
    1. This class is associated with fitting the dataset to the dataloaders 
    2. Attributes: 
      * dataset - DataFrame 
      * tokenizer - tokenizer 
      * max_len - number - (default - 256) 
      * input_attr - string - DataFrame field for input text
      * target_attr - string - DataFrame field for output class label 
    3. The dataloaders contain the following: 
      * sentences - the main input text (one among args, args+prompts, prompts)
      * ids 
      * attention mask 
      * token type ids 
      * target - the label associated with it ( numerical format and not as str)
    """
    def __init__(self, dataset, tokenizer, max_len, input_attr, target_attr):
        self.tokenizer = tokenizer
        self.data = dataset
        self.text = dataset[input_attr]
        self.targets = dataset[target_attr]
        self.max_len = max_len
       

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        

        return {
            'sentence': text,
            
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

class RobertaClass(torch.nn.Module):
    """
    1. Class associated with the model - RoBERTa Large  
    2. Attributes: 
      * output_params - number - (number of prediction classes) 
      * model_name - string  
    """
    def __init__(self, output_params, model_name):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, output_params)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
   
        return output


def calcuate_accu(big_idx, targets):
    """
    Function to calculate Accuracy
    """
   
    n_correct = (big_idx==targets).sum().item()
    return n_correct

def generate_classification_report(preds, targets, unique_labels): 
    """
    Function to generate the classification report
    """

    print(classification_report(targets, preds, target_names=unique_labels, digits=4))
    cm = confusion_matrix(targets, preds) 
    print("Confusion Matrix: ")
    print(cm)
  
    print("Per class Accuracy: ", cm.diagonal()/cm.sum(axis=1) ) 


def predict(loader, model_path, output_params, model_name,device_no, epochs=1):
    """
    Attributes:
    1. loader - LogicalFallacy 
    2. model_path - string - path to the model for eval 
    3. epoch - number - set to a default 1. 
    4. 
    """
    val, og_val = [], [] 
    model = torch.load(model_path)
    torch.cuda.set_device(device_no)
    torch.cuda.empty_cache() 
    model.to(device)
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    test_answers = [[[],[]], [[],[]]]

    n_correct = 0 
    nb_tr_steps = 0 
    nb_tr_examples = 0 
    for epoch in range(epochs):
        for steps, data in tqdm(enumerate(loader, 0)):
            sentence = data['sentence']
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
        
            outputs = model.forward(ids, mask, token_type_ids)
            #print(torch.max(outputs.data, dim=1))
            _, max_indices = torch.max(outputs.data, dim=1)
            
            val.extend(max_indices.tolist())   
            
           
            og_val.extend(targets.tolist())
            
            n_correct+= calcuate_accu(max_indices, targets) 

            nb_tr_steps +=1 
            nb_tr_examples+=targets.size(0)
            
           
    accuracy = (n_correct*100)/nb_tr_examples 

        
    return accuracy, val, og_val
                                                                
def get_unique_labels(dataset, class_label):
  unique_labels = dataset[class_label].unique().tolist() 
  return unique_labels

       
def driver_code(test_file, model_path, model_name, input_attr,  target_attr, tokenizer_name, class_label, output_params, device_no, max_len=256, valid_batch_size=4):
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name, truncation=True, do_lower_case=True)

    print("------Reading Test File------")
    test_df = pd.read_csv(test_file)
    unique_labels = get_unique_labels(test_df, class_label)
    print("------Tokenizing Test Data------")
    test_set = LogicalFallacy(test_df, tokenizer, max_len, input_attr, target_attr)

    test_params = {'batch_size': valid_batch_size,
                'shuffle': True,
                'num_workers': 0
                }
    
    test_loader = DataLoader(test_set, **test_params)

   
    print("------Running Evaluation------")
   
    value, preds, targets = predict(test_loader, model_path, output_params, tokenizer_name, device_no)
    print("\n\n\n")
    print("Model Name: ", model_name)
    print("Accuracy of the model: ", value) 
    print("Classification Report: ")
    generate_classification_report(preds, targets, unique_labels)
    
if __name__ == '__main__': 
  parser = argparse.ArgumentParser()
  parser.add_argument('-ts', '--test_file', help = "Path to the test file") 
  parser.add_argument('-mn', '--model_name', help="Name of the model") 
  parser.add_argument('-mp', '--model_path', help="Path to the saved model") 
  parser.add_argument('-tk', '--tokenizer', help="Name of the tokenizer") 
  parser.add_argument('-ia', '--input_attr', help="Input text field name") 
  parser.add_argument('-ta', '--target_attr', help="Target label field") 
  parser.add_argument('-cl', '--class_label', help="Target class label")
  parser.add_argument('-d', '--device', help="device number", default=0)
  parser.add_argument('-op', '--output_params_num', help="Number of prediction class labels")
  args = parser.parse_args()

  driver_code(
    args.test_file, 
    args.model_path, 
    args.model_name, 
    args.input_attr, 
    args.target_attr,
    args.tokenizer,
    args.class_label,
    int(args.output_params_num), 
    int(args.device)
  )

