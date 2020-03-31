import os
from subprocess import Popen
import sys

module_path = os.path.abspath(os.path.join('..', '..', os.getcwd()))
print(f"In extract_contexual_relevance_model_features, {module_path}")
sys.path.append(module_path)

print(f"cwd: {os.getcwd()}")

scripts_path = 'extract_dataset_with_feats'

init_program = 'initialize_training_dataset.py'

feature_extraction_script_names = ['extract_count_features.py',
                'extract_language_model_feats.py',
               'extract_labels.py',
               'extract_relatedness_features.py',
                'extract_yap_features.py']

run_merge_programs = True

merge_programs = ['merge_extracted_feats_and_labels.py']

init_cmd = f'python {scripts_path + os.sep + init_program}'
print(f"Running init cmd: {init_cmd}...")
os.system(init_cmd)

print(f"Running concurrent feature extraction of contextual relevance model...")

all_processes = []

def run_process(cmd):
    print(f"Running {cmd}")
    p = Popen(cmd, shell=True)
    all_processes.append(p)

for s_name in feature_extraction_script_names:
    script_full_path = scripts_path + os.sep + s_name
    cmd = f'python {script_full_path}'
    run_process(cmd)

print(f"Waiting for concurrent programs...\n\n")

for p in all_processes: p.wait()

print(f"Concurrent programs done.\n\n")

if run_merge_programs:
    print('running merge & all labels programs...\n\n')
    merge_cmds = [f'python {scripts_path + os.sep + fname}' for fname in merge_programs]
    for cmd in merge_cmds:
        print(f"Running {cmd}")
        os.system(cmd)

print(f"Finished running all programs")