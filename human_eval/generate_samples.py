from human_eval.data import write_jsonl, read_problems
from octoai_inf import OctoAIEndpointLM
import json
import itertools
import re

NUM_RUNS=5

MODEL_NAMES = [
    "codellama-7b-instruct-fp16",
    "codellama-13b-instruct-fp16",
    "codellama-34b-instruct-int4",
    "codellama-34b-instruct-fp16",

    "llama-2-13b-chat-fp16",
    "llama-2-70b-chat-fp16",
    "llama-2-70b-chat-int4",
    "mistral-7b-instruct-fp16"

]
MODEL_NAMES_PROD = [
    "codellama-7b-instruct-fp16",
    "codellama-13b-instruct-fp16",
    "codellama-34b-instruct-int4",
    "codellama-34b-instruct-fp16",

    "llama-2-13b-chat-fp16",
    "llama-2-70b-chat-fp16",
    "llama-2-70b-chat-int4",
    "mistral-7b-instruct-fp16"
]



def extract_output(output):
    out = output.split("[/INST]")[-1].split("</s>")[0].strip()
    out = out.replace('```', '')
    return out


def extract_code(string):
    out = string.replace('```', '')
    try:
        code_idx_start = out.index('[PYTHON]')
        out = "\n" + out[code_idx_start + len('[PYTHON]'):]
        code_idx_end = out.index('[/PYTHON]')
        out = out[:code_idx_end]
    except ValueError:
        return out
    except IndexError:
        return out
    return extract_output(out)

def remove_empty_lines(code):
    lines = code.split('\n')
    non_empty_lines = [line for line in lines if line.strip() != '']
    return '\n'.join(non_empty_lines)

def extract_code_and_imports_as_string(llama_output):
    llama_output = llama_output.replace('```', '')
    llama_output = remove_empty_lines(llama_output)
    blocks = re.split(r'\n\n|\n(?=def |class )', llama_output)
    code_blocks = []
    imports = []

    for block in blocks:
        lines = block.split('\n')
        current_code_block = []
        in_code_block = False

        for line in lines:
            line_stripped = line.strip()

            if line_stripped.startswith('import ') or line_stripped.startswith('from '):
                imports.append(line_stripped)

            if line_stripped.startswith('def ') or line_stripped.startswith('class '):
                # Start of a new code block
                in_code_block = True
                current_code_block.append(line)

            elif in_code_block:
                # Collect lines belonging to the current code block
                current_code_block.append(line)

        # Add the current code block if present
        if current_code_block:
            code_blocks.append('\n'.join(current_code_block))

    code_blocks_str = "\n\n".join(code_blocks) if code_blocks else ""
    imports_str = "\n".join(imports) if imports else ""
    try:
        code_idx_end = [i for i in range(len(code_blocks_str)) if code_blocks_str.startswith('    return', i)][-1]
        while (code_blocks_str[code_idx_end] != '\n') and (code_idx_end < len(code_blocks_str) - 1):
            code_idx_end += 1
        code_blocks_str = code_blocks_str[:code_idx_end + 1]
    except IndexError:
        pass

    if imports_str and code_blocks_str:
        return f"\n{imports_str}\n\n\n{code_blocks_str}"
    elif imports_str:
        return f"Imports:\n{imports_str}"
    elif code_blocks_str:
        return f"\n{code_blocks_str}"
    else:
        return "No code blocks found in the response."

def extract_code_llama(string):
    out = string.replace('```', '')
    try:
        code_idx_start = out.index('\n')
    except ValueError:
        return out
    if "def " not in out[:code_idx_start]:
        out = out[code_idx_start:]
    out = out.replace("python", "")
    try:
        code_idx_end = [i for i in range(len(out)) if out.startswith('    return', i)][-1]
        while (out[code_idx_end] != '\n') and (code_idx_end < len(out) - 1):
            code_idx_end += 1
        return out[:code_idx_end + 1]        
    except IndexError:
        return out
    

def generate_one_completion(inp,):
    tmp = endpoint.call_octoai_inference(inp)
    tmp = tmp['choices'][0]['message']['content'] 
    print("MODEL OUT:   \n\n")
    print(tmp)
    out = extract_code_and_imports_as_string(tmp)
    print("PARSING ::   \n\n")
    print(out)
    return out

def get_requests(problems):
    requests = []
    for task_id in problems:
        requests.append(problems[task_id]["prompt"].strip())
    return requests


def generate_completions(inps):
    endpoint = OctoAIEndpointLM(MODEL_NAMES[0], batch_size=16)
    results = endpoint._model_generate_parallel(inps)
    return results
    

problems = read_problems()
import time
import os
import tqdm
for platform in ['dev', 'prod']:
    mnms = MODEL_NAMES_PROD if platform == 'prod' else MODEL_NAMES
    for model_name in mnms:
        os.mkdir(f"results_{platform}/{model_name}/")
        endpoint = OctoAIEndpointLM(model_name, use_prod=True if platform=='prod' else False)

        for i in range(NUM_RUNS):
            samples = []
            num_samples_per_task = 1
            gen_start_time = time.time()
            samples = [
                dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
                for task_id in tqdm.tqdm(problems)
            ]
            gen_end_time = time.time()
            
            os.mkdir(f"results_{platform}/{model_name}/{model_name}_{i}")
            write_jsonl(f"results_{platform}/{model_name}/{model_name}_{i}/samples.jsonl", samples)
            tmp = os.popen(f"evaluate_functional_correctness results_{platform}/{model_name}/{model_name}_{i}/samples.jsonl").read()
            with open(f"results_{platform}/{model_name}/{model_name}_{i}/time_stat.txt", 'w') as f:
                f.write(str((gen_end_time - gen_start_time) / 60))
                f.write(tmp.split('\n')[-2])
                