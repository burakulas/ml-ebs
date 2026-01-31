#!/usr/bin/env python3

import subprocess
import sys

scripts = ["1_prepare_training_data.py", "2_extract_training_features.py", "3_train_models.py"]

for script in scripts:
    #print(f"--- Running {script} ---")
    result = subprocess.run([sys.executable, script])
    
    if result.returncode != 0:
        print(f"!!! Error in {script}. Stopping sequence. !!!")
        break
    
    # This prints only after the script is done
    #print(f">>> {script} finished successfully.\n") 

print("All scripts completed.")
