import json
import sys
def get_samples(filename):
    with open(filename, "r") as fp:
        for line in fp:
            if any(not x.isspace() for x in line):
                yield json.loads(line)
model_path = sys.argv[1] 
for sample, problem in zip(get_samples(model_path + "samples.jsonl_results.jsonl"), get_samples("data/problem_samples.jsonl")):
    if not sample['passed']:
        test_example = ""
        print("question: ")
        print(problem['prompt'])
        print()
        print()
        print(sample['completion'])
        print(sample['result'])
        print(problem["test"])
        entry_point = problem["entry_point"]
        if "[PYTHON]" in sample["completion"]:
            import pdb
            pdb.set_trace()
        test_example += problem["test"] + "\n" + sample["completion"]  + "\n" + f"check({entry_point})"
        with open(f"test_scripts_llama/{entry_point}.py", 'w') as f:
            f.write(test_example)
        print("=====================================")