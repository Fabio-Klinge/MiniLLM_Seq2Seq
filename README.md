# Knowledge Distillation using reverse KL divergence for Natural Language Inference
This is an extension of [MiniLLM](https://github.com/microsoft/LMOps/tree/main/minillm). Using the existing code-base to apply it to Sequence to Sequence models for a NLI task


## 1 Environment

---
The environment can be installed using:

```bash
bash install.sh
```

(Cuda 11.4 is needed)


## 2 Data
### 2.1 Resources
The ANLI dataset can be downloaded here: [HF-ANLI](https://huggingface.co/datasets/facebook/anli/tree/main)

### 2.2 Data Processing
To preprocess the raw dataset from huggingface you can use 
```bash
bash t5-scripts/tools/process_data_anli.sh /PATH_TO/MINILLM
```
This will write all the contents of the downloaded *.parquet files into a .txt and use it to create a binarized, indexed and memory mapped dataset as well as a json. 

 ### 2.2 Dataset Augmentation
 This work added reason fields, where missing, to the ANLI dataset. If one is interested in the resulting dataset feel free to contact me.

 For using the functionallity one can use:
 ```bash
bash t5-scripts/sft_base.sh /PATH_TO/MINILLM
```
with the arguement "--refinent" set. Prompts in the processed. Prompts in the preprocessed dataset to be changed respectively.



## 3 Models
### 3.1 Resources
+ The official checkpoints of the Flan-T5 model family serve as starting models for this work and can be found here: [Flan-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5)

#### Base Pre-trained Models
To run supervised fine-tuning you can do:
```bash
bash t5-scripts/sft_base.sh /PATH_TO/MINILLM
```


You can also change the `CKPT` variable in each script to the corresponding model name to enable Transformers to download checkpoints automatically. 

## 4 Run Evaluation
```bash
bash t-5-scripts/eval/evaluate_acc_models.sh /PATH_TO/MINILLM
```

## 5 Train
To run the knowledge distillation two checkpoints mus be present.

### 5.1 Baselines
The final checkpoints are selected by the Accuracy on ANLI. The script used in 3.1 can be used to finetune multiple models.
### 5.2 MiniLLM
#### Train
```bash
bash t-5-scripts/minillm/train_large_large.sh /PATH_TO/LMOps/minillm
```
The script can also be used to train other model sizes but it is recommended to adjust hyperparameter an deepspeed config for that case.
## 6 Results
Results can be found in the respective folder. If one is interested in the resulted model checkpoints or the augmented ANLI dataset feel free to contact me: fklinge@uni-osnabrueck.de.



