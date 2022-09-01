# Scripts

This folder contains the scripts for running the RoBERTa baseline used for fallacy identification at the coarse-grained and fine-grained level 

## For Training

```bash
python train_roBERTa.py 
       -tr /path/to/training_dataset 
       -dv /path/to/validation_dataset 
       -tk 'name_of_the_tokenizer' ('roberta-large') 
       -mn 'name_of_the_model' ('roberta-large') 
       -mp /path/to/save_the_trained_model 
       -ia 'input_attribute' 
       -ta 'target_attribute' 
       -d  CUDA_device
       -op number_of_prediction_class_labels 
       -ep number_of_epochs_for_training  

```

## For evaluation `eval_RoBERTa.py` 
```bash 
python eval_roBERTa.py 
      -ts  /path/to/test_dataset 
      -mn  name_of_the_model
      -mp  /path/to/saved_trained_model 
      -tk  name_of_the_tokenizer  
      -ia  input_attribute 
      -ta  target_attribute 
      -cl  class_label
      -d   CUDA_device
      -op  number_of_prediction_class_labels 
``` 

### Attributes description 

`-ts`:path to the test file 
    * subroutinization turned on  
    * strict error-reporting  
    * GlyphOrderAndAliasDB is applied 

`-mn`: name of the model
     * Usually its roberta-base 
       
`-mp`:path to the trained model 

`-tk`: name of the tokenizer 
     * Usually its roberta-base
       
`-ia`: input attribute name
     * Input text attribute field. 
     * Could be either among 'source_article' (sentence), 'clean_prompt' (sentence + prompt), 'cleaner_prompt' (prompt) depending on which input the model has been trained over. 
       
`-ta`: target attribute name
     * Usually, this is the numerical label value assigned to the classes that are being classified. 
     * Its the 'label' field in the dataset ( for binary classification for the fine-grained classifier) or the 'mapped_label' (as in the coarse-grained classifier) 

`-cl`: the class attribute name 
     * This is the name of the attribute that constitutes to the string value of the classes of fallacy 
     * Its the 'updated_label' ( for binary classification) or the 'broad_class' for the (coarse-grained classifier)
       
 `-d`: the CUDA device number to mount the loaded model onto. 
 
 `-op`: The number of prediction classes that are being dealt with in classification. 
      * Its is 2 for fine-grained classification and 3 for the coarse-grained classification 
      
  `-r`: release mode  
    * subroutinization turned on  
    * strict error-reporting  
    * GlyphOrderAndAliasDB is applied   

