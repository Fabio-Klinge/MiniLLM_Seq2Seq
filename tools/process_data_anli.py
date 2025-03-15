# For def parquet_to_txt()
from datasets import load_dataset
import pandas as pd
import os
import re
from glob import glob
import multiprocessing
import time
import torch
import json
import sys
import numpy as np
from data_utils.indexed_dataset import make_builder
from transformers import T5Tokenizer
from arguments import get_args
import sys



class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        Encoder.tokenizer = T5Tokenizer.from_pretrained(self.args.model_path, legacy=False)

    def encode(self, line):
        line = json.loads(line)

        label_map = {
            0: "entailment",
            1: "neutral",
            2: "contradiction"
        }

        #template = "This is a Natural Language Inference Task. Decide whether the relationship between hypothesis and premise is classified as entailment, contradiction or neutral. provide an explanation for your classification. hypothesis: {hypothesis} premise: {premise}"

        #template = ("{context}\n\nBased on the paragraph above can we conclude that \"{hypothesis}\"? Explain why.\n\n{options}")
        template = ("{context}\nCan we infer the following and why?\n{hypothesis}\n\n{options}")

        options = "OPTIONS:\n- entailment\n- contradiction\n- neutral"
        # Create dataset to refine the models created reasons
        if self.args.refinement:
            reason = line.get('reason', '')
            print("og reason", reason)

            
            label = line.get('label', '')
            premise = line.get('premise', '')
            
        
            # If reason is provided, add it to the label
            if len(reason) != 0:  
                # template = (
                #     "Evaluate the following reason for this nli example. If the reason is repetitive, corrupted or wrong improve these shortcomings. Return an empty string if you are unsure! If the reason is valid, return the exact reason you were given. Premise: {premise} Hypothesis: {hypothesis} Label: {label} Reason: {reason}"
                # )
                # template = """Improve NLI explanation:
                # Premise: {premise}
                # Hypothesis: {hypothesis}
                # Label: {label}
                # Original: {reason}

                # # Output: """

                # Default to Neutral if label is not in map
                #label = label_map.get(label, "neutral") 

                #  sun´bsteps, [1], "output, conclusion, entailment...", \nowiki, if now point remove until the whole thing, not sentence
                # br> in premise

                ### Clean generated reasons ###

                # Check for complete sentence, remove incomplete ends
                if reason[-1] != ".":
                    print("point accessed")
                    last_period_index = reason.rfind(".")
                    if last_period_index != -1:
                        reason = reason[:last_period_index + 1]  # Include the period (+1)

                if any(bad_word in reason.lower() for bad_word in ["nowiki", "nli", "[i]", "[ii]", "[iii]", "[1] and [2] and [3]"]):
                    print("badbad words accessed")
                    reason = ""

                # Remove bad starts
                for bad_start in ["hypothesis:","output:", "conclusion:", "entailment:", "contradiction:", "neutral:"]:
                    print("bad start accessed", bad_start)
                    if bad_start in reason.lower():
                        # Find the position of "hypothesis:" (case insensitive)
                        position = reason.lower().find(bad_start)
                        # Return the string after the word
                        reason = reason[position + len(bad_start):].strip()

                for bad_word in ["substeps", "br>"]:
                    if bad_word in reason.lower():
                        reason = reason.replace(bad_word, "")

                for bad_word in ["substeps", "br>"]:
                    if bad_word in premise.lower():
                        premise = premise.replace(bad_word, "")

                
                
                
                # Check for repeated words and remove them from dataset
                word_repetitions = []
                words = reason.lower().split()
                for i in range(len(words) - 1):
                    if words[i] == words[i + 1]:
                        word_repetitions.append(words[i])
                if len(word_repetitions) > 5:
                    print("repeated words accessed")
                    reason = ""
                
                # Check for repeated sentences 
                sentences = [s.strip() for s in reason.split('.') if s.strip()]
                sentence_repetitions = []
                for i in range(len(sentences) - 1):
                    if sentences[i] == sentences[i + 1]:
                        sentence_repetitions.append(sentences[i])
                if len(sentence_repetitions) > 1:
                    print("repeated sentences accessed")
                    reason = ""

                # Remove lines with empty reasons
                if len(reason) < 5:
                    print("short reason accessed, none returned")
                    return None

                print("new reason", reason)             

                print(f"label", {label}, {type(label)}, flush=True)
                class_label = Encoder.tokenizer.encode(label, add_special_tokens=True, truncation=True, max_length=self.args.max_prompt_length)
                label = f"{str(label)}:{str(reason)}"
                #prompt = template.format(premise=line["premise"], hypothesis=line["hypothesis"])

                prompt = template.format(context=line["premise"], hypothesis=line["hypothesis"], options=options)


                print("prompt", prompt)





                # Save for dataset creation
                uid = Encoder.tokenizer.encode(line["uid"], add_special_tokens=False, truncation=True, max_length=self.args.max_prompt_length)
                premise = Encoder.tokenizer.encode(line["premise"], add_special_tokens=False, truncation=True, max_length=self.args.max_prompt_length)
                hypothesis = Encoder.tokenizer.encode(line["hypothesis"], add_special_tokens=False, truncation=True, max_length=self.args.max_prompt_length)
            



                prompt_tokens = Encoder.tokenizer.encode(prompt, add_special_tokens=False, truncation=True, max_length=self.args.max_prompt_length)
                #### Add second eos token for multiprocessing (removed when returned)
                #full_tokens = Encoder.tokenizer.encode(prompt + label, add_special_tokens=True, truncation=True, max_length=self.args.max_length)
                label_tokens = Encoder.tokenizer.encode(label, add_special_tokens=True, truncation=True, max_length=self.args.max_prompt_length)
                #label_tokens = full_tokens[len(prompt_tokens):]
                #return line, prompt, label, prompt_tokens, (label_tokens[:-2] + label_tokens[-1:]), len(line)








                return line, prompt, label, prompt_tokens, label_tokens, len(line), uid, premise, hypothesis, class_label



            # discard lines without reason
            else:
                return None


        else:

            # template = (
            #     "Your task is to determine whether hypothesis entails, contradicts, or is neutral to the premise."
            #     "The first word of your output needs to be either \"entailment\", \"Contradiction\" or \"neutral\". Followed by an explanation for your answer.  \n\n"
            #     "### Premise:\n{premise} \n Hypothesis: {hypothesis} \n ### Classification & Reason:\n"
            # )

            # template = (
            #     "Your task is to classify the relationship between the premise and the hypothesis. Further provide a rationale for your classification."
            #     "The first word of your output needs to be one of the classifications (\"Entailment\", \"Contradiction\" or \"Neutral\"). Followed by an detailed but concise explanation. Your output should be in the following format: \n \"{{Classification}}: {{An explanation for the classification}} .  \n\n"
            #     "### Premise:\n{premise} \n Hypothesis: {hypothesis} \n ### Your Classification & Reason:\n"
            # )

            # template = (
            #     "Classify the relationship between the premise and the hypothesis. Further provide a rationale for your classification. "
            #     "The first word of your output needs to be one of the classifications (\"Entailment\", \"Contradiction\" or \"Neutral\") followed by a concise, logical explanation.\n\n"
            #     "Your output should be in the following format:\n"
            #     "Classification <extra_id_0> An explanation for the classification.\n\n"
            #     "### Premise:\n{premise}\n\n"
            #     "### Hypothesis:\n{hypothesis}\n\n"
            #     "### Your Classification & Reason:\n"
            # )

            # template = ("Classify the relationship between the premise and the hypothesis and provide a rationale explaining your classification. The classifications are \"Entailment\", \"Contradiction\" or \"Neutral\".\n"
            #             "These two examples show how the task works and how to structure your output:\n\n"

            #             "Example 1:\n"
            #             "### Premise: The man is running.The street is wide.\n"
            #             "### Hypothesis: A person is moving.\n"
            #             "### Your Classification & Reason:\n"
            #             "Entailment <extra_id-0> running implies movement.\n\n"
            #             "Example 2:\n"
            #             "### Premise: The dog is black sitting on a white floor in the sun.\n"
            #             "### Hypothesis: The animal is white.\n"
            #             "### Your Classification & Reason:\n"
            #             "Contradiction: Black and white are opposite colors.\n"
            #             "Now classify this example:\n"
            #             "### Premise: {premise}\n"
            #             "### Hypothesis: {hypothesis}\n"
            #             "### Your Classification & Reason:\n")

            #template = ("mnli please explain why hypothesis:{hypothesis} is classified as {label} for premise:{premise}")
            #template = ("explain why the relationship between the premise and the hypothesis is classified as {label}. premise:{premise} hypothesis:{hypothesis}")
            
            #used for 1st datset iter
            #template = ("mnli please explain why hypothesis:{hypothesis} is classified as {label} for premise:{premise}")





            label = line.get('label', '')
            label = label_map.get(label, "neutral") 

            class_label = Encoder.tokenizer.encode(label, add_special_tokens=True, truncation=True, max_length=self.args.max_prompt_length)
            # Default to Neutral if label is not in map

            prompt = template.format(context=line["premise"], hypothesis=line["hypothesis"], options=" * ".join(options))

            # Save for dataset creation
            uid = Encoder.tokenizer.encode(line["uid"], add_special_tokens=False, truncation=True, max_length=self.args.max_prompt_length)
            premise = Encoder.tokenizer.encode(line["premise"], add_special_tokens=False, truncation=True, max_length=self.args.max_prompt_length)
            hypothesis = Encoder.tokenizer.encode(line["hypothesis"], add_special_tokens=False, truncation=True, max_length=self.args.max_prompt_length)
        
        
            # If reason is provided, add it to the label
            if len(line.get('reason', '')) != 0:  
                reason = line.get('reason', '')
                label = f"{label}:{reason}"

                prompt_tokens = Encoder.tokenizer.encode(prompt, add_special_tokens=False, truncation=True, max_length=self.args.max_prompt_length)
                #### Add second eos token for multiprocessing (removed when returned)
                #full_tokens = Encoder.tokenizer.encode(prompt + label, add_special_tokens=True, truncation=True, max_length=self.args.max_length)
                label_tokens = Encoder.tokenizer.encode(label, add_special_tokens=True, truncation=True, max_length=self.args.max_prompt_length)
                #label_tokens = full_tokens[len(prompt_tokens):]
                #return line, prompt, label, prompt_tokens, (label_tokens[:-2] + label_tokens[-1:]), len(line)


            # Add padding token where reason is missing (neglected in loss calculation)
            else:
                # Add a default explanation if missing
                label = f"{label}"

                prompt_tokens = Encoder.tokenizer.encode(prompt, add_special_tokens=False, truncation=True, max_length=self.args.max_prompt_length)
                # Add second eos token for multiprocessing (removed when returned)
                #full_tokens = Encoder.tokenizer.encode(prompt + label, add_special_tokens=True, truncation=True, max_length=self.args.max_length)
                label_tokens = Encoder.tokenizer.encode(label, add_special_tokens=True, truncation=True, max_length=self.args.max_prompt_length) #+ [Encoder.tokenizer.eos_token_id]
        
            return line, prompt, label, prompt_tokens, label_tokens, len(line), uid, premise, hypothesis, class_label


def parquet_to_txt(args):
    """
    Saves all .parquet files from ANLI to a data.txt file. 
    Splits and round are marked by  e.g. <test_r1> for test round 1.
    Each line in the .txt file is a JSON object containing the 'uid', 'premise', 'hypothesis', 'label', and 'reason' fields. 
    """
    print(args.data_dir)
    # Define paths to the ANLI .parquet files
    anli_parquet_files = glob(f"{args.data_dir}plain_text/*.parquet")
    print(f"{args.data_dir}plain_text/*.parquet")

    os.makedirs(args.data_dir, exist_ok=True)

    num = 0
    # Write .parquet files to .txt
    with open(os.path.join(args.data_dir, "data.txt"), "w") as f:
        for file_path in anli_parquet_files:
            # Extract split and round from the filename
            parts = file_path.split('/')[-1].split('-')
            round_split = parts[0]  # e.g., 'train_r1'
            marker = f"<{round_split}>"

            # Write the marker to the file
            f.write(marker + "\n")

            df = pd.read_parquet(file_path)
            for index, row in df.iterrows():
                premise = re.sub(r"\n+", " ", row['premise'])
                hypothesis = re.sub(r"\n+", " ", row['hypothesis'])
                label = row['label']
                reason = re.sub(r"\n+", " ", row['reason']) if row['reason'] else ""

                # Format the data as desired. For example, you can use JSON-like formatting
                formatted_data = {
                    "uid": row['uid'],
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "label": label,
                    "reason": reason
                }

                # Write the formatted data as a JSON string
                f.write(f"{json.dumps(formatted_data)}\n")
                num += 1

    print("Number of lines:", num)

def main():
    try:
        print("OK")
        args = get_args()
        print("### Test Dataset is set to", args.small_dataset)

        # Create data.txt from .parquet files
        parquet_to_txt(args)

        if args.small_dataset:
            raw_data = []
            with open(os.path.join(args.data_dir, "data.txt")) as f:
                for i in range(10):
                    line = f.readline()
                    if not line:
                        break
                    raw_data.append(line.strip())
        else:
            # Load each line
            with open(os.path.join(args.data_dir, "data.txt")) as f:
                raw_data = f.readlines()



        # Split data according to rounds and splits
        rounds = {'r1': {'train': [], 'dev': [], 'test': []},
                'r2': {'train': [], 'dev': [], 'test': []},
                'r3': {'train': [], 'dev': [], 'test': []}}

        # Add generated train data to the training data
        if args.refinement:
            # Pre-load training data from your files
            with open(os.path.join(args.base_path, "train_r1/refined_data.txt")) as f:
                rounds['r1']['train'] = [line.strip() for line in f.readlines()]
            with open(os.path.join(args.base_path, "train_r2/refined_data.txt")) as f:
                rounds['r2']['train'] = [line.strip() for line in f.readlines()]
            with open(os.path.join(args.base_path, "train_r3/refined_data.txt")) as f:
                rounds['r3']['train'] = [line.strip() for line in f.readlines()]

            current_split = None 
            for line in raw_data:
                # Remove any leading/trailing whitespace
                line = line.strip()  
                # Skip all train sections entirely
                if line.startswith("<train_r1>") or line.startswith("<train_r2>") or line.startswith("<train_r3>"):
                    current_split = None  # Set to None to ignore lines until next marker
                    continue
                elif line.startswith("<dev_r1>"):
                    current_split = ('r1', 'dev')
                    continue
                elif line.startswith("<test_r1>"):
                    current_split = ('r1', 'test')
                    continue
                elif line.startswith("<dev_r2>"):
                    current_split = ('r2', 'dev')
                    continue
                elif line.startswith("<test_r2>"):
                    current_split = ('r2', 'test')
                    continue
                elif line.startswith("<dev_r3>"):
                    current_split = ('r3', 'dev')
                    continue
                elif line.startswith("<test_r3>"):
                    current_split = ('r3', 'test')
                    continue
                else:
                    if current_split:  # Ensure there is a current split defined
                        rounds[current_split[0]][current_split[1]].append(line)

        # Files from .parquet file solely (original ANLI)
        else:
            current_split = None
            for line in raw_data:
                # Remove any leading/trailing whitespace
                line = line.strip()
                
            # Round 1 sections
                if line.startswith("<train_r1>"):
                    current_split = ('r1', 'train')
                    continue
                elif line.startswith("<dev_r1>"):
                    current_split = ('r1', 'dev')
                    continue
                elif line.startswith("<test_r1>"):
                    current_split = ('r1', 'test')
                    continue
                # Round 2 sections
                elif line.startswith("<train_r2>"):
                    current_split = ('r2', 'train')
                    continue
                elif line.startswith("<dev_r2>"):
                    current_split = ('r2', 'dev')
                    continue
                elif line.startswith("<test_r2>"):
                    current_split = ('r2', 'test')
                    continue
                # Round 3 sections
                elif line.startswith("<train_r3>"):
                    current_split = ('r3', 'train')
                    continue
                elif line.startswith("<dev_r3>"):
                    current_split = ('r3', 'dev')
                    continue
                elif line.startswith("<test_r3>"):
                    current_split = ('r3', 'test')
                    continue
                else:
                    if current_split:  # Ensure there is a current split defined
                        rounds[current_split[0]][current_split[1]].append(line)

        # Small dataset for testing
        line_counter = 0
        small_dataset_complete = False

        # Process data
        for round in rounds:
            for split in rounds[round]:
                split_data = rounds[round][split]
                if not split_data:
                    continue
                
                # split_dir = os.path.join(args.processed_data_dir, round, split)
                # os.makedirs(split_dir, exist_ok=True)
                # Determine the directory based on split
                if split == 'dev':
                    split_dir = os.path.join(args.processed_data_dir, round, 'train')
                else:
                    split_dir = os.path.join(args.processed_data_dir, round, split)
                os.makedirs(split_dir, exist_ok=True)

                if split == 'dev' or split == 'test':
                    args.refinement = False
                else:
                    args.refinement = True

                encoder = Encoder(args)
                pool = multiprocessing.Pool(processes=args.data_process_workers, initializer=encoder.initializer)
                # Endode() gets inputs from multiprocessing pool
                encoded_docs = pool.imap_unordered(encoder.encode, split_data, chunksize=50)
                # Get rid of lines without reason
                encoded_docs = [doc for doc in encoded_docs if doc is not None]
                proc_start = time.time()
                total_bytes_processed = 0

                bin_file = os.path.join(split_dir, f"{split}_0.bin")
                idx_file = os.path.join(split_dir, f"{split}_0.idx")
                binary_builder = make_builder(bin_file, impl="mmap", dtype=np.uint16)

                inst_num = 0
                print("#" * 10, f"{round} {split}", "#" * 10)

                prompt_lens = []
                label_lens = []

                json_file = open(os.path.join(split_dir, f"{split}.jsonl"), "w")

                for lid, (line, prompt_str, label_str, prompt, label, bytes_processed, int_uid, int_premise, int_hypothesis, class_label) in enumerate(encoded_docs):

                    total_bytes_processed += bytes_processed

                    binary_builder.add_item(torch.IntTensor(prompt + [-1] + label + [-1] + int_uid + [-1] + int_premise + [-1] + int_hypothesis + [-1] + class_label))

                    json_file.write(json.dumps({
                        "uid": line["uid"],
                        "prompt": prompt_str,
                        "label": label_str,
                        "prompt_tokens": prompt,
                        "label_tokens": label,
                    }) + "\n")

                    prompt_lens.append(len(prompt))
                    label_lens.append(len(label))

                    inst_num += 1

                    line_counter += 1
                    if bool(args.small_dataset) and line_counter >= args.small_dataset:
                        small_dataset_complete = True

                    if lid % 1000 == 0:
                        current = time.time()   
                        elapsed = current - proc_start
                        mbs = total_bytes_processed / elapsed / 1024 / 1024
                        print(f"Processed {lid} documents. {inst_num} instances.",
                            f"({lid/elapsed} docs/s, {mbs} MB/s).",
                            file=sys.stderr)
                        

                if not small_dataset_complete:
                    binary_builder.finalize(idx_file)
                    pool.close()
                    json_file.close()

                if small_dataset_complete:
                    binary_builder.finalize(idx_file)
                    pool.close()
                    json_file.close()
                    break

                

                print("Data num", len(prompt_lens))
                print("Prompt lengths.", "Mean:", np.mean(prompt_lens), "Max:", np.max(prompt_lens), "Min:", np.min(prompt_lens))
                print("label", "Mean:", np.mean(label_lens), "Max:", np.max(label_lens), "Min:", np.min(label_lens))

    except Exception as e:
        with open("error_log.txt", "w") as f:
            f.write(f"Error: {str(e)}\n")
            f.write(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':    
    main()