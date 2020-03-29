import os
import json

ora = r"E:\mdtel_data\data\manual_labeled_v2\doccano\depression_ora_export.json"
yoav = r"E:\mdtel_data\data\manual_labeled_v2\doccano\depression_yoav_export.json"

with open(ora, encoding='utf-8') as f:
    ora_lines = [json.loads(x) for x in f.readlines()]

with open(yoav, encoding='utf-8') as f:
    yoav_lines = [json.loads(x) for x in f.readlines()]


def get_ora_line(line_text):
    for l2 in ora_lines:
        if line_text == l2['text']:
            found_line = True
            ora_annotations = l2['annotations']
            if ora_annotations != []:
                ann = ora_annotations[0]
                print("ora", line_text[ann['start_offset']:ann['end_offset']], ann['label'])
            return ora_annotations
    return None

line_none_count = 0
found_anns = 0

yoav_disorder_number = 11
yoav_chemical_number = 12
ora_disorder_number = 15
ora_chemical_number = 16

for line_idx, line in enumerate(yoav_lines):
    line_text = line['text']
    if line['annotations'] != []:
        ann = line['annotations'][0]
        print(line_text[ann['start_offset']:ann['end_offset']], ann['label'])
    ora_annotations = get_ora_line(line_text)
    if ora_annotations == None:
        line_none_count += 1
    if ora_annotations and ora_annotations != []:
        for n in ora_annotations:
            if n['label'] == ora_chemical_number:
                n['label'] = yoav_chemical_number
            if n['label'] == ora_disorder_number:
                n['label'] = yoav_disorder_number
        found_anns += 1
        line['annotations'] += ora_annotations

p = r"E:\mdtel_data\data\manual_labeled_v2\doccano\depression_export.json"
with open(p, 'w', encoding='utf-8') as f:
    for l in yoav_lines:
        f.write(json.dumps(l, ensure_ascii=False) + "\n")

print(f"line_none_count: {line_none_count}, found_anns: {found_anns}, yoav: {len(yoav_lines)}, ora: {len(ora_lines)}")



print("Yo")