
import os
tmp = os.listdir("test_scripts_llama")
for script in tmp:
    file = open(f"test_scripts_llama/{script}").read()
    print(file)
    os.system(f"python test_scripts_llama/{script}")
    import pdb
    pdb.set_trace()