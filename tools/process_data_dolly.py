import multiprocessing
import os
import time
import torch
import json
import sys
from numerize.numerize import numerize
import numpy as np
from data_utils.indexed_dataset import make_builder
from transformers import AutoTokenizer, 
from arguments import get_args


# 1. Implement an Encoder, which gives it a line of input data and it returns you the tokenized result.
class Encoder(object): 
    def __init__(self, args):
        self.args = args
#DLINK=$(echo -n "aHR0cHM6Ly9jb252ZXJzYXRpb25odWIuYmxvYi5jb3JlLndpbmRvd3MubmV0L2JlaXQtc2hhcmUtcHVibGljL01pbmlMTE0vZ3B0Mi50YXI/c3Y9MjAyMy0wMS0wMyZzdD0yMDI0LTA0LTEwVDEzJTNBMTElM0E0NFomc2U9MjA1MC0wNC0xMVQxMyUzQTExJTNBMDBaJnNyPWMmc3A9ciZzaWc9NGNYSklqVlJaSElCV3FIalBnRG4lMkYwMW9jenBEV1hpcG1QQ1VrM1o4dmJRJTNE" | base64 --decode)
#wget -O gpt2.tarDLINK=$(echo -n "aHR0cHM6Ly9jb252ZXJzYXRpb25odWIuYmxvYi5jb3JlLndpbmRvd3MubmV0L2JlaXQtc2hhcmUtcHVibGljL01pbmlMTE0vZ3B0Mi50YXI/c3Y9MjAyMy0wMS0wMyZzdD0yMDI0LTA0LTEwVDEzJTNBMTElM0E0NFomc2U9MjA1MC0wNC0xMVQxMyUzQTExJTNBMDBaJnNyPWMmc3A9ciZzaWc9NGNYSklqVlJaSElCV3FIalBnRG4lMkYwMW9jenBEV1hpcG1QQ1VrM1o4dmJRJTNE" | base64 --decode) $DLINK
    def initializer(self):
        Encoder.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)

    def encode(self, line):
        line = json.loads(line)
        if len(line["input"]) == 0:
            template = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:\n"
            )
            prompt = template.format(instruction=line["instruction"])
        else:
            template = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
            )
            # Add additional information to input field
            prompt = template.format(instruction=line["instruction"], input=line["input"])
            
        response = line["output"]
        prompt_tokens = Encoder.tokenizer.encode(prompt, add_special_tokens=False)
        full_tokens = Encoder.tokenizer.encode(prompt + response, add_special_tokens=False) + [Encoder.tokenizer.eos_token_id]
        response_tokens = full_tokens[len(prompt_tokens):]
        
        if len(prompt_tokens) > self.args.max_prompt_length:
            return None, None, None, None, len(line)
        
        return line, prompt, prompt_tokens, response_tokens, len(line)


def main():
    print("OK")
    args = get_args()
        
    args.processed_data_dir = os.path.join(args.processed_data_dir, args.model_type)

    os.makedirs(args.processed_data_dir, exist_ok=True)
    
    with open(os.path.join(args.data_dir, "raw.jsonl")) as f:
        raw_data = f.readlines()

    if args.dev_num > 0:
        all_data = {
            "valid": raw_data[:args.dev_num],
            "train": raw_data[args.dev_num:]
        }
    else:
        all_data = {
            "train": raw_data
        }
    
    for split in all_data:
        
        # encoder use the tokenizer to encode data
        encoder = Encoder(args)

        # 2. Mapping all datas with Encoder, with the help of multiprocessing
        pool = multiprocessing.Pool(processes=args.data_process_workers, initializer=encoder.initializer)
        encoded_docs = pool.imap_unordered(encoder.encode, all_data[split], chunksize=50)
        proc_start = time.time()
        total_bytes_processed = 0
        
        bin_file = os.path.join(args.processed_data_dir, f"{split}_{0}.bin")
        idx_file = os.path.join(args.processed_data_dir, f"{split}_{0}.idx")

        binary_builder = make_builder(bin_file, impl="mmap", dtype=np.uint16)
        
        # put tokenized data into binary_builder
        inst_num = 0
        print("#"*10, split, "#"*10)
        
        prompt_lens = []
        response_lens = []
        
        json_file = open(os.path.join(args.processed_data_dir, f"{split}.jsonl"), "w")
        
        for lid, (line, prompt_str, prompt, response, bytes_processed) in enumerate(encoded_docs):
            total_bytes_processed += bytes_processed
            if prompt is None:
                continue
            
            if args.only_prompt:
                if len(prompt) < args.max_length:
                    binary_builder.add_item(torch.IntTensor(prompt))
                else:
                    continue
            else:
                binary_builder.add_item(torch.IntTensor(prompt + [-1] + response))

            json_file.write(json.dumps({
                "instruction": line["instruction"],
                "prompt": prompt_str,
                "input": line["input"],
                "output": line["output"],
            }) + "\n")

            prompt_lens.append(len(prompt))
            response_lens.append(len(response))

            inst_num += 1
            if lid % 1000 == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                print(f"Processed {lid} documents. {inst_num} instances.",
                    f"({lid/elapsed} docs/s, {mbs} MB/s).",
                    file=sys.stderr)

        # finish compressing tokenized data into `bin_file`, and generate meta information into `idx_file`
        binary_builder.finalize(idx_file)

        # close multiproceessing mapping
        pool.close()
        json_file.close()
                
        print("Data num", len(prompt_lens))
        print("Prompt lengths.", "Mean:", np.mean(prompt_lens), "Max:", np.max(prompt_lens), "Min:", np.min(prompt_lens))
        print("Response", "Mean:", np.mean(response_lens), "Max:", np.max(response_lens), "Min:", np.min(response_lens))


if __name__ == '__main__':
    main()