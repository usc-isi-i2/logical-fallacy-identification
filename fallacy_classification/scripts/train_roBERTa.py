import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
import numpy as np
from tqdm import tqdm
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
        self.l1 = RobertaModel.from_pretrained(model_name)
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


def train_loop(train_loader, dev_loader, model_path,  model_name, device=0, output_params=3, epochs=5, learning_rate=1e-05):
  """
  Attributes: 
  * train_loader - LogicalFallacy dict
  * test_loader - LogicalFallacy dict 
  * model_path - string - to save the trained model 
  * device - number - CUDA device to mount the model for training 
  * model_name - string 
  * output_params - number - number of output classes for prediction 
  * epochs - number 
  * learning_rate - float 
  """
  train_loss = []
  dev_loss = []
  train_accuracy = []
  dev_accuracy = []
  model = RobertaClass(output_params, model_name)
  model.to(device)
  loss_function = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(params =  model.parameters(), lr=learning_rate)

  dev_answers = [[[],[]], [[],[]]]
  for epoch in range(epochs):
    for phase in ['Train', 'Val']:
      if(phase == 'Train'):
        model.train()
        loader = train_loader
      else:
        model.eval()
        loader = dev_loader  
      epoch_loss = 0
      epoch_acc = 0
     
      for steps, data in tqdm(enumerate(loader, 0)):
        sentence = data['sentence']
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)
      
        outputs = model.forward(ids, mask, token_type_ids)

        loss = loss_function(outputs, targets)        
        
        epoch_loss += loss.detach()
        _, max_indices = torch.max(outputs.data, dim=1)
        bath_acc = (max_indices==targets).sum().item()/targets.size(0)
        epoch_acc += bath_acc

        if (phase == 'Train'):
          train_loss.append(loss.detach()) 
          train_accuracy.append(bath_acc)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
        else:
          dev_loss.append(loss.detach()) 
          dev_accuracy.append(bath_acc)
         
      print("Phase: ", phase)
      print(f"{phase} Loss: {epoch_loss/steps}")
      print(f"{phase} Accuracy: {epoch_acc/steps}")
  
  torch.save(model, model_path)
  

def driver_code(train_file, 
                dev_file, 
                tokenizer_name, 
                model_name, 
                model_path, 
                input_attr, 
                target_attr, 
                device_num=0, 
                output_params_num=3, 
                total_epochs=5, 
                learning_rate_val=1e-05, 
                max_len=256, 
                train_batch_size=8, 
                valid_batch_size=4):
    """
    Attributes: 
    1. train_file - string - path to the training file (csv) 
    2. dev_file - string - path to the dev file (csv) 
    3. model_name - string - name of the RoBERTa model used 
    4. model_path - string - path to save the finetuned model 
    5. input_attr - string - name of the input attribute in the dataframe ( text ) used for classification 
    6. target_attr - string - name of the target attribute in the dataframe ( class label) used for classificaiton 
    7. device_num - number - CUDA device number for mounting the model 
    8. output_params_num - number - number of output class for prediction 
    9. total_epochs - number - total number of training epochs 
    10. learning_rate_val - float 
    11. max_len - number - max_len of tokens 
    12. train_batch_size - number 
    13. valid_batch_size - number 
    """

    #Opening the training 
    train_df = pd.read_csv(train_file)
    dev_df = pd.read_csv(dev_file)

    #Loading the Tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name, truncation=True, do_lower_case=True)


    #Preprocessing steps of tokenization and creating the data loaders for the sets of data
    train_set = LogicalFallacy(train_df, tokenizer, max_len, input_attr, target_attr)
    dev_set = LogicalFallacy(dev_df, tokenizer, max_len, input_attr, target_attr)

    train_params = {'batch_size': train_batch_size,
                'shuffle': True,
                'num_workers': 0
                }

    dev_params = {'batch_size': valid_batch_size,
                'shuffle': True,
                'num_workers': 0
                }
    train_loader = DataLoader(train_set, **train_params)
    dev_loader = DataLoader(dev_set, **dev_params)

    #Training 
    train_loop( train_loader, dev_loader, model_path, model_name, device=device_num, output_params=output_params_num, epochs=total_epochs, learning_rate=learning_rate_val)
    

if __name__ == "__main__": 
  parser = argparse.ArgumentParser() 
  parser.add_argument('-tr', '--train_file', help="Training File") 
  parser.add_argument('-dv', '--dev_file', help='Dev File') 
  parser.add_argument('-tk', '--tokenizer', help="Name of the Tokenizer") 
  parser.add_argument('-mn', '--model_name', help="Model Name") 
  parser.add_argument('-mp', '--model_path', help="Path to Save the model") 
  parser.add_argument('-ia', '--input_attr', help="Input text field name") 
  parser.add_argument('-ta', '--target_attr', help="Target Attribute field")
  parser.add_argument('-d', '--device', help="device number", default=0)
  parser.add_argument('-op', '--output_params_num', help="Number of prediction class labels")
  parser.add_argument('-ep', '--epoch', help='Number of epochs', default=5) 
  args = parser.parse_args()



  driver_code(args.train_file, args.dev_file, args.tokenizer, args.model_name, args.model_path, args.input_attr, args.target_attr, args.device, args.output_params_num, args.epoch)
