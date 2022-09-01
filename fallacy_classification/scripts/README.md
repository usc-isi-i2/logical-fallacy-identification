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

## For evaluation 
```bash 
python eval_roBERTa.py 
      -ts /path/to/test_dataset 
      -mn 'name_of_the_model' ( 'roberta-large' ) 
      -mp /path/to/saved_trained_model 
      -tk 'name_of_the_tokenizer' ('roberta-large') 
      -ia 'input_attribute' 
      -ta 'target_attribute' 
      -d   CUDA_device
      -op  number_of_prediction_class_labels 
