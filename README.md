# Knowledge Distillation using reverse KL divergence for Natural Language Inference
This is an extension of [MiniLLM]{https://github.com/microsoft/LMOps/tree/main/minillm}. Using the existing code-base to apply it to Sequence to Sequence models for a NLI task


## 1 Environment

---
The environment can be installed using:

```bash
bash install.sh
```

(Cuda 11.4 is needed)


## 2 Data
### 2.1 Resources
The ANLI dataset can be downloaded here: [HF-ANLI]{https://huggingface.co/datasets/facebook/anli/tree/main}

### 2.2 Data Processing
To preprocess the raw dataset from huggingface you can use 
```bash
bash t5-scripts/tools/process_data_anli.sh /PATH_TO/MINILLM
```

This will write all the contents of the downloaded *.parquet files into a .txt and use it to create a binarized, indexed and memory mapped dataset as well as a json. 

## 3 Models
### 3.1 Resources
+ The official checkpoints of the Flan-T5 model family serve as starting models for this work and can be found here: [Flan-T5]{https://huggingface.co/docs/transformers/model_doc/flan-t5}
#### Base Pre-trained Models
To run supervised fine-tuning you can:
```bash
bash t5-scripts/sft- /PATH_TO/MINILLM
```


Alternatively, you can also change the `CKPT` variable in each script to the corresponding model name to enable Transformers to download the base models automatically. For example, set `CKPT="gpt2-large"` in `scripts/gpt2/sft/sft_large.sh` causes download of the gpt2-large base model from the HugginFace model hub.

## 4 Run Evaluation
```bash
bash scripts/gpt2/eval/run_eval.sh /PATH_TO/LMOps/minillm
bash scripts/opt/eval/run_eval.sh /PATH_TO/LMOps/minillm
bash scripts/llama/eval/run_eval.sh /PATH_TO/LMOps/minillm
```

## 5 Train
We provide example commands for GPT-2 models. Similar scripts for model families can be found in `scripts/opt` and `scripts/llama`. All our experiments are conducted on 16 \* 32V100, which can be reduced for small models.
Some large models require tensor parallel size = 4, which is set in the scripts with `--model-parallel` and `--model-parallel-size` options.

### 5.1 Baselines
The final checkpoints are selected by the Rouge-L scores.
#### Fine-tune the teacher models
```bash
bash scripts/gpt2/sft/sft_xlarge.sh /PATH_TO/LMOps/minillm
```
#### SFT Baselines
```bash
bash scripts/gpt2/sft/sft_base.sh /PATH_TO/LMOps/minillm
bash scripts/gpt2/sft/sft_medium.sh /PATH_TO/LMOps/minillm
bash scripts/gpt2/sft/sft_large.sh /PATH_TO/LMOps/minillm
```

#### KD Baselines
```bash
bash scripts/gpt2/kd/kd_base.sh /PATH_TO/LMOps/minillm
bash scripts/gpt2/kd/kd_medium.sh /PATH_TO/LMOps/minillm
bash scripts/gpt2/kd/kd_large.sh /PATH_TO/LMOps/minillm
```

#### SeqKD Baselines
Generate and process responses with the teacher:
```bash
bash scripts/gpt2/tools/generate_data_seqkd.sh /PATH_TO/LMOps/minillm
bash scripts/gpt2/tools/process_pseudo_data_seqkd.sh /PATH_TO/LMOps/minillm
```
Fine-tune the model with SeqKD:
```bash
bash scripts/gpt2/seqkd/seqkd_base.sh /PATH_TO/LMOps/minillm
bash scripts/gpt2/seqkd/seqkd_medium.sh /PATH_TO/LMOps/minillm
bash scripts/gpt2/seqkd/seqkd_large.sh /PATH_TO/LMOps/minillm
```

### 5.2 MiniLLM
#### Initial Checkpoints
The final checkpoints are selected by the **validation loss**.
```bash
bash scripts/gpt2/sft/sft_base.sh /PATH_TO/LMOps/minillm
bash scripts/gpt2/sft/sft_medium.sh /PATH_TO/LMOps/minillm
bash scripts/gpt2/sft/sft_large.sh /PATH_TO/LMOps/minillm
```

#### Train
The final checkpoints are selected by the Rouge-L scores.
```bash
bash scripts/gpt2/minillm/train_base_xl.sh /PATH_TO/LMOps/minillm
bash scripts/gpt2/minillm/train_medium_xl.sh /PATH_TO/LMOps/minillm
bash scripts/gpt2/minillm/train_large_xl.sh /PATH_TO/LMOps/minillm
```

For the data we use:
+ `PROMPT_DATA_DIR` is the SFT data ($\mathcal{D}$, Dolly), which is required.
+ `LM_DATA_DIR` is the plain-text corpus ($\mathcal{D}_\text{PT}$), which is optional. See `minillm/scripts/gpt2/minillm/train_base_xl_no_pt.sh` for training without `LM_DATA_DIR` (by just commenting out the `OPTS+=" --lm-data-dir ${LM_DATA_DIR}"` line).

### 5.3 Multi-Node training
Multi-Node training is launched by `deepspeed`. We provide an example script in `scripts/llama/sft/sft_7B_mn.sh` for multi-node training. Compared to single-node scripts, some of the `DISTRIBUTED_ARGS` are changed, and you need to specify a hostfile like `configs/hostfiles/node_0_1` to tell the script which nodes to use. For more information, please refer to HuggingFace's [tutorial](https://huggingface.co/docs/transformers/main_classes/deepspeed#the-deepspeed-launcher).


## 6 Citation
```bibtex
@inproceedings{minillm,
  title={MiniLLM: Knowledge Distillation of Large Language Models},
  author={Gu, Yuxian and Dong, Li and Wei, Furu and Huang, Minlie},
  booktitle={Proceedings of ICLR},
  year={2024}
}
