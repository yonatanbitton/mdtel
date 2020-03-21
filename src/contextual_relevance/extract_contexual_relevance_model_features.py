import os
from subprocess import Popen

scripts_path = 'extract_dataset_with_feats'

init_program = 'initialize_training_dataset.py'

feature_extraction_script_names = ['extract_count_features.py',
                'extract_language_model_feats.py',
                'extract_relatedness_features.py',
                'extract_yap_features.py']

merge_programs = ['merge_extracted_feats.py',
                'add_labels_doccano_by_offset.py']


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

print(f"Concurrent programs done, running merge & all labels programs...\n\n")

merge_cmds = [f'python {scripts_path + os.sep + fname}' for fname in merge_programs]
for cmd in merge_cmds:
    print(f"Running {cmd}")
    os.system(cmd)

print(f"Finished running all programs")