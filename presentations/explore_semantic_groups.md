```python
import os
import sys
import pandas as pd

path = r"E:\mdtel_data\data\high_recall_matcher\HEB_TO_ENG_DISORDERS_CHEMICALS.csv"
```


```python
df = pd.read_csv(path)
```

# Disorders

1. Acquired Abnormality  
2. Anatomical Abnormality  
3. Cell or Molecular Dysfunction  
4. Congenital Abnormality  
5. Disease or Syndrome  
6. Experimental Model of Disease  
7. Finding  
8. Injury or Poisoning  
9. Mental or Behavioral Dysfunction  
10. Neoplastic Process  
11. Pathologic Function  
12. Sign or Symptom

## 1. Acquired Abnormality


```python
print(list(df[df['STY'] == 'Acquired Abnormality']['STR'].sample(100)))
```

    ['scabs', 'Facies Leonine', 'crusts', 'superficial ulcers', 'cysts vallecular', 'bladder diverticulum', 'Cubitus varus', 'hyperplasia denture', 'Clawed toes', 'flat foot', 'neurotrophic ulcer', 'Jejunal ulcer', 'Colostomy Stoma', 'Ureterocele', 'bursa cyst', 'Prolasso uterovaginale', 'Spinal Stenosis', "Baker's Cyst", 'intestinal adhesions', 'thoracic fistula', "Baker's Cyst", 'joint mouse', 'Acquired Phimosis', 'Anal fistula', 'herniated disk', 'Acquired scoliosis', 'effusions subdural', 'glandular cyst', 'amputation stump', 'Acquired spondylolysis', 'Postural lordosis', 'Fibroatheroma', 'sores gum', "baker's cysts", 'loss teeth', 'simplex lentigo', 'adhesions bowel', 'obliquities pelvic', 'szurkehalyog', 'Epulis Fissuratum', 'false passage', "Baker's cyst", 'simplex lentigo', 'Acquired Meningomyelocele', 'curly toe', 'bone bruises', 'Claw toe', 'j-pouch', 'Short thighs', 'foot callus', 'perianal hematoma', 'Acquired lordosis', 'Rectocele', 'heal ulcer', 'Ureteric Fistula', 'Septal hypertrophy', 'tonsil ulcers', 'peritoneum adhesions', 'hernias irreducible', 'camptocormia', 'loss dental', 'Striae', 'burns scars', 'Follicular cyst', 'gouty tophus', 'jaw cysts', 'Claw feet', 'Colostomia', 'Bent spine', 'Urethral Caruncle', 'hernia femoral', 'Umbilicated nodule', 'adhesions abdominal', 'corns toes', 'bone bruises', 'Tooth loss', 'Duodenal scar', 'phlebolithiasis', 'cataract unilateral', 'prolapse vaginal', 'Stump', 'damages muscles', 'Anus Prolapse', 'soft cataract', 'Indolent ulcer', 'Peritoneal adhesions', 'Dental Fissure', 'Unspecified cataract', 'Thin-cap fibroatheroma', 'Hypertrophic Scar', 'body joints', 'hypertrophic scar', 'Macular pucker', 'Keratosis palmaris', 'retinal cyst', 'Gingival cysts', 'av graft', 'Stapes fixation', 'Stretch marks', 'subchondral cyst']
    

## 2. Anatomical Abnormality


```python
print(list(df[df['STY'] == 'Anatomical Abnormality']['STR'].sample(100)))
```

    ['Intestinal Polyps', 'microgenia', 'deformity gibbus', 'Perineural Cyst', 'Colon diverticula', 'Malar flattening', 'Talon cusp', 'deformities vaginal', 'fistula', 'Cockleshell ear', 'hematocyst', 'Mandibular retrusion', 'Sigmoidovaginal fistula', 'anorectal malformation', 'Urethra Cyst', 'Infundibular Cyst', 'Jejunal fistula', 'Deep bite', 'biliary stricture', 'traumatic aneurysm', 'Wide ulna', 'oral fistulas', 'pes cavus', 'Muscle Diastasis', 'invert nipples', 'Nasal malformation', 'cyst genital', 'Supernumerary nostrils', 'Back knee', 'tracheo-esophageal fistula', 'foot deformities', 'Ectopic scrotum', 'Mandibular hyperplasia', 'cysts urethral', 'Overjet', 'Rib abnormalities', 'clubbed toes', 'Parathyroid Cyst', 'Fistola duodenale', 'Short eyelid', 'gingival fibromatosis', 'club toe', 'deformity spine', 'Enthesis abnormality', 'hyperdistension', 'hematocyst', 'flat feet', 'Bone cysts', 'Anterior openbite', 'Distichiasis', 'nephrogenic rests', 'Absent canines', 'Fistule oesophagienne', 'Midnasal atresia', 'Juvenile Cataract', 'Polygonal-shaped calices', 'Epidermoid Cyst', 'Ischiorectal fistula', 'stricture ureter', 'Positional Plagiocephaly', 'cerebral herniation', 'esophageal fistula', 'Retroflexed Uterus', 'Straight clavicles', 'Eyebrow abnormalities', 'Head abnormalities', 'knees knocked', 'Vascular Fistula', 'Miliary Aneurysm', 'diverticulum colon', 'Urethral Fistula', 'Central polydactyly', 'bone cyst', 'Abnormal thrombosis', 'Ventral hernia', 'stomach fistula', 'Solitary Cyst', 'Elfin ear', 'fistula fecal', 'perineal fistula', 'bronchoesophageal fistula', 'Bone deformities', 'Flat acetabulum', 'vascular malformations', 'Premaxillary excess', 'Laryngeal abnormalities', 'Long chin', 'stomach diverticulum', 'Genu recurvata', 'knee knocked', 'Vesicocervicovaginal fistula', 'bladder cyst', 'Mandibular Retroposition', 'abnormalities nails', 'Ureteral stenosis', 'keratinous cyst', 'Hypoplastic antitragus', 'cysts inclusion', 'mouth polyps', 'Genu valga']
    

## 3. Cell or Molecular Dysfunction


```python
print(list(df[df['STY'] == 'Cell or Molecular Dysfunction']['STR'].sample(100)))
```

    ['Double Minutes', 'Aneuploid', 'koilocytosis', 'Atypical Platelet', 'reciprocal translocation', 'T-cell dysfunction', 'inflammatory atypia', 'mild dysplasia', 'nonsense mutation', 'cemental dysplasia', 'Nonsense mutation', 'axotomy response', 'Neurodegeneration', 'hypersegmented neutrophil', 'Cytogenetic Aberration', 'uniparental disomy', 'Chromosome Anomalies', 'Chromosomal Fragility', 'somatic mutation', 'atypia inflammatory', 'Transversion Mutations', 'Chromosome breakage', 'atypical cells', 'cabots rings', 'Base-Base Mismatch', 'new mutation', 'Chromosomal Duplication', 'Retrograde Degeneration', 'cytopathic effect', 'Mixed Mutations', 'Pericentric Inversion', 'novel mutation', 'Nissl Degeneration', 'Monoallelic Mutation', 'Chromosomal Instability', 'degeneration nerves', 'Deletion Mutation', 'Genomic Instability', 'Deletion Mutation', 'Base-Base Mismatch', 'abnormal platelets', 'Chromosome Alterations', 'Paracentric Inversion', 'Gene Deregulation', 'Oncogene Activation', 'Synthetic Lethality', 'Chromosome instability', 'cytopathic effect', 'Proteostasis Deficiencies', 'Mitochondrial Swelling', 'Uniparental Isodisomy', 'Somatic mutation', 'Microsatellite Instability', 'genetic translocations', 'cells damage', 'abnormal chromosomes', 'Tetraploidy', 'chromosome abnormalities', 'aneuploidies', 'Near-Haploidy', 'abnormal platelet', 'Cytogenetic Aberration', 'Duplication', 'point mutation', 'Robertsonian Translocation', 'Intergenic Mutation', 'chromosomal aberrations', 'Hypotriploidy', 'cell degeneration', 'Glandular Metaplasia', 'Nuclear alterations', 'Transversion Mutations', 'Elongation Mutation', 'pathologic autolysis', 'Point Mutation', 'translocation robertsonian', 'Abnormal Platelet', 'Preneoplastic Change', 'chromosomal aberrations', 'Abnormal meiosis', 'Protein Truncation', 'Koilocytotic Atypia', 'Genetic Change', 'Chromosome inversion', 'dyskaryosis', 'Atypical Platelet', 'Abnormal platelets', 'Somatic Mutation', 'Protein Truncation', 'Chromosome Inversion', 'chromosomal translocation', 'Inversion', 'cytologic atypia', 'transneuronal degeneration', 'abnormal platelet', 'Severe Dysplasia', 'Intergenic Variant', 'Transversion', 'Read-Through Mutation', 'degeneration secondary']
    

## 4. Congenital Abnormality


```python
print(list(df[df['STY'] == 'Congenital Abnormality']['STR'].sample(100)))
```

    ['Dysplastic kidney', 'Brainstem dysplasia', 'missing ribs', 'Trigonocephaly', 'tarsal coalitions', 'microcolon', 'Ulegyria', 'Nevus Flammeus', 'bite scissors', 'Ectopic teeth', 'Limb amelia', "peters' anomaly", 'Microcorneas', 'hare lip', 'chordee', 'Congenital epiblepharon', 'Megalocornea', 'Cor Triatriatum', 'Synophrys', 'divisum pancreas', 'Anomalies confenitales', 'fallots tetralogy', 'pelvic kidney', 'Amyoplasia', 'Undescended testicles', 'Crisscross Heart', 'anomaly peter', 'microsomia hemifacial', 'hydrocele testicle', "Roger's disease", 'Missing eyelids', 'Hypospadie', 'supernumerary ovary', 'macrodactyly', 'Jejunal Atresia', 'Imperforate rectum', 'roger syndrome', 'anus imperforate', 'Polycoria', 'Periauricular earpits', 'congenital neuropathy', 'Spinal Dysraphism', 'Glandular hypospadias', 'primordial dwarf', 'Partial Ankyloglossia', 'Acheiropodia', 'Pectus Carinatum', 'Cranium bifidum', 'spinal meningocele', 'hemicrania', 'Hemivertebrae', 'Fundus Albipunctatus', 'Underdeveloped scrotum', 'Short radii', 'Goniodysgenesis', 'Renal dysplasia', 'neurenteric cyst', 'Open spine', 'abnormality jaw', 'solitary kidney', 'monorchidism', 'Temporal hypotrichosis', 'Pigeon chest', 'Branchioma', 'Microglossia', 'Short thumb', 'acystia', 'Congenital Spondylolysis', 'Macrodactylia', 'Biliary Atresia', 'Finger-like thumbs', 'Deformities', 'Fromont Anomaly', 'canal atrioventricular', 'Cleft Jaw', 'anomaly heart', 'Goodman syndrome', 'dermal sinus', 'Cloacogenic bladder', 'Bronchial atresia', 'Micrencephaly', "Sprengel's deformity", 'Coledococele', 'Congenital blindness', 'kidney malrotation', 'Mandible small', 'Sagittal craniosynostosis', 'veleszuletett rendellenesseg', 'Posterior wedging', 'Ipospadia', 'carolis syndrome', 'extra fingers', 'Corneolenticular adhesion', 'Narrow mouth', 'Ocular melanocytosis', 'trigonocephalus', 'Madelung deformity', 'Congenital alopecia', 'fused kidney', 'Meckel diverticulum']
    

## 5. Disease or Syndrome


```python
print(list(df[df['STY'] == 'Disease or Syndrome']['STR'].sample(100)))
```

    ['infarcts pulmonary', 'myocardium disorder', 'labyrinthine disorders', 'Tropical Sprue', 'Cutaneous osteosis', 'Hypertrophic osteoarthropathy', 'romberg disease', 'sleep talking', 'excessive sleep', 'Ankylosed teeth', 'syndrome nephritic', 'mps iii', 'Cerebellotrigeminal-dermal dysplasia', 'Cardiomyopathic Lentiginosis', 'Non-Ketotic Hyperglycinemia', 'Chorioretinal dystrophy', 'Balantidium Infection', 'Kraepelin disease', 'Otospongiosis', 'infection minor', 'Quinapril pseudoallergy', 'cardiac disorder', 'Endocrinopathy', 'labirintitis', 'Prekallikrein Deficiency', 'blood disorder', 'cysts sublingual', 'Bezoar disorder', 'sigmoiditis', 'organic disorders', "hurler's syndrome", 'enteric fever', 'tracheal disease', 'Struma lymphomatosa', 'acne conglobate', 'spinal arthritis', 'Thromboangitis Obliterans', 'Crystalline Arthropathies', 'rubella complications', 'gender disorder', 'Pendular Nystagmus', 'pneumothorax tension', 'Giardiasis', 'Dandy-Walker anomaly', 'auras visual', 'abdominal disorder', 'protozoal meningoencephalitis', 'Dolore mestruale', 'Alpha-methylacetoaceticaciduria', 'Enkopresis', 'spleen disorder', 'Status epilepticus', 'Brain Hamartoma', 'pentosuria', 'Hallux Rigidus', 'allergies seafood', 'Temporal arteritis', 'syndromes werner', 'Allergic bronchitis', 'Eczematoid dermatitis', 'headaches cluster', "Morgellon's", 'erlichiosis', 'Ectopic Ureter', 'Comitant strabismus', 'Squint', 'Metal fever', 'Cystoliths', 'Trench fever', 'Mesenteric lipodystrophy', 'Telangiectatica Congenita', 'Scarring alopecia', 'Clostridium Infections', 'Zellweger Syndrome', 'verruca acuminata', 'epidemic pleurodynia', 'syndrome wells', 'fluid uterus', 'skin infection', "Evans' Syndrome", 'Hairy elbows', 'raynauds phenomenon', 'Menopausal Syndrome', 'aorta disorder', 'Environmental Allergy', 'Dermatosis herpetiformis', 'enamel hypoplasia', 'Airway Obstruction', 'osteochondroses', 'Egg Allergy', 'pharyngeal spasm', 'subendocardial ischemia', 'Glucosylceramide Lipidosis', "horton's arteritis", 'Perioral dermatitis', 'Hyperphenylalaninemia', 'Hepatite virale', 'Fatco Syndrome', 'Familial hypercholesterolemia', 'tooth erosion']
    

## 6. Experimental Model of Disease


```python
print(list(df[df['STY'] == 'Experimental Model of Disease']['STR'].sample(100)))
```

    ['Transgenic Model', 'disorder model', 'Knock-out', 'Rous sarcoma', 'Knock-out', 'Adjuvant Arthritis', 'Allergic encephalomyelitis', 'Patient-Derived Xenograft', 'Experimental Tumor', 'Adjuvant Arthritis', 'autoimmune encephalomyelitis', 'Knockout', 'Experimental Hepatomas', 'Avian Sarcoma', 'Rous sarcoma', 'adjuvant arthritis', 'avian sarcoma', 'disease models', 'Experimental Parkinsonism', 'Harding-Passey Melanoma', 'Mouse Glucagonoma', 'arthritis collagen', 'Experimental Melanomas', 'Allergic Encephalomyelitis', 'Cancer Model', 'collagen-induced arthritis', 'Disease model', 'Streptozotocin Diabetes', 'adjuvant arthritis', 'Experimental Neoplasms', 'collagen-induced arthritis', 'Tissue Model', 'Experimental Tumor', 'Experimental Parkinsonism', 'Xenograft Model', 'mouse model', 'Experimental Parkinsonism', 'Mouse Model', 'Streptozocin Diabetes', 'Experimental Myasthenia', 'Cancer Model', 'autoimmune encephalomyelitis', 'Mouse Model', 'Experimental Melanoma', 'Experimental Leukemia', 'Xenograft Model', 'ascites tumor', 'Allergic Encephalomyelitis', 'Experimental Sarcomas', 'Experimental Leukemia', 'Experimental Leukemias', 'disease model', 'Allergic Encephalomyelitis', 'adjuvant disease', 'Harding-Passey Melanoma', 'Xenograft Model', 'collagen-induced arthritis', 'Experimental Melanomas', 'collagen arthritis', 'Experimental Sarcoma', 'mouse model', 'Transgenic Model', 'collagen arthritis', 'Mouse Glucagonoma', 'Experimental Hepatoma', 'Tissue Model', 'Rous Sarcoma', 'Experimental Sarcomas', 'Knockout', 'arthritis collagen', 'adjuvant disease', 'Rous Sarcoma', 'Knockout', 'arthritis collagen', 'Rous Sarcoma', 'adjuvant arthritis', 'Collagen Arthritis', 'avian sarcoma', 'Experimental Myasthenia', 'Allergic encephalomyelitis', 'Streptozotocin Diabetes', 'ascites tumor', 'Non-Rodent Model', 'disorder model', 'Non-Rodent Model', 'Experimental Melanoma', 'Transgenic Model', 'Experimental Hepatomas', 'Tissue Model', 'Streptozocin Diabetes', 'Experimental Leukemias', 'Alloxan Diabetes', 'Mouse Model', 'Ascites tumors', 'Patient-Derived Xenograft', 'mouse model', 'Harding-Passey Melanoma', 'Mouse Glucagonoma', 'Experimental Tumor', 'Experimental Neoplasms']
    

## 7. Finding


```python
print(list(df[df['STY'] == 'Finding']['STR'].sample(100)))
```

    ['Lymphoma Spread', 'Vermis hypoplasia', 'Gray sclerae', 'pulmonary granuloma', 'testicle mass', 'Therapy-Related Toxicity', 'Stoma ulcer', 'Rhizomelic humeri', 'Short carpals', 'Small caudate', 'Constricted ear', 'ventricular gallop', 'Hypoargininemia', 'frothy stool', 'cyst wrist', 'Small head', 'intellect impaired', 'neck tremor', 'gynecological problem', 'gallbladder distend', 'diadochokinesis', 'Delayed eruption', 'Angor animi', 'Immature genitalia', 'Coxa deformity', 'fluids disorder', 'prosecution', 'Lips full', 'Paced Rhythm', 'Subcutaneous emphysema', 'overactivity', 'Fetal macrosomia', 'Pharyngeal Reflex', 'High Cholesterol', 'Extensor posturing', 'Laryngeal weakness', 'roth spot', 'soft stools', 'Thin build', 'Nostril notching', 'patients problems', 'Necrotic Change', 'painful orgasm', 'Broad cranium', 'Flared nostrils', 'bullying school', 'increasing lactation', 'maternity leave', 'induced labour', 'muscle splinting', 'diaphoresis', 'Pregnancy Status', 'Skeletal Maturity', 'wheelchair transfers', 'Device Fell', 'Duodenal bands', 'Annulled', 'urea high', 'lesions vulva', 'skipped heartbeats', 'Brachymelic dwarfism', 'alerts', 'Mediastinal Shift', 'left handedness', 'normal pth', 'Hair abnormality', 'Occasional dribbling', 'molding', 'Double antitragus', 'Sweating', 'weepiness', 'Intervention Required', 'Asperger-like features', 'nail ridging', 'Defective ejaculation', 'referral sources', 'Remission', 'Thinness', 'Short digit', 'Somatic mosaicism', 'education find', 'Big maxilla', 'breath holding', 'Finger hypermobility', 'Tubular proteinuria', 'Pseudoacinar formation', 'unifocal pvc', 'Incomplete Ossification', 'tonsils swollen', 'hypothermia', 'Preoccupation finding', 'fast state', 'hyperplasia cervical', 'Flat ears', 'bladder scarring', 'Joint hyperlaxity', 'Decreased Apgar', 'Neuroendocrine Differentiation', 'Petechies gastriques', 'post ptca']
    

## 8. Injury or Poisoning


```python
print(list(df[df['STY'] == 'Injury or Poisoning']['STR'].sample(100)))
```

    ['pugilistica dementia', 'bruises contusions', 'blisters throat', 'venous puncture', 'colon injuries', 'contusion brain', 'neck sprain', 'Andere fracturen', 'injury ankle', 'chest wounds', 'Physical Violence', 'patellar fracture', 'poisoning fish', 'Sports Injuries', 'stomach injury', 'sting plant', 'sting fish', 'open fractures', 'alcohol poison', 'Hip subluxation', 'wrist sprain', 'Scalding injury', 'burning vagina', 'aortic injury', 'minamata disease', 'buttocks trauma', 'trachea injury', 'fractures sternum', 'animal scratch', 'burns scalp', 'Lead poisoning', 'blisters fracture', 'Brain concussion', 'thallium poison', 'leg strain', 'poison ammonia', 'Accidental hypothermia', 'burns injuries', 'Self-mutilation', 'contusions multiple', 'esophageal injury', 'obstetrical paralysis', 'skin burn', 'Uveal Prolapse', 'critical incidents', 'back burns', 'eye injuries', 'concussions', 'Unspecified privation', 'alveolar fractures', 'injuries scrotum', 'radiation necrosis', 'strain tendons', 'hip strain', 'injuries leg', 'bite wounds', 'phenobarbital toxicity', 'burn scald', 'knife wounds', 'fracture dislocations', 'exposures heat', 'Traumatic Arthropathy', 'Forehead Trauma', 'teflon poisoning', 'chlorate poisoning', 'Trauma acustico', 'bruised', 'lacerations skin', 'coccyx injury', 'encephalopathy toxic', 'injuries urethra', 'cat bite', 'subtrochanteric fracture', 'hypervitaminosis d', 'electric injuries', 'fractures metacarpal', 'bites blackfly', 'barbiturate overdose', 'patellar dislocations', 'Electric Injuries', 'leg bruise', 'birthing trauma', 'fractures neck', 'cervical strain', 'Fractura femur', 'groin injury', 'Neck Sprain', 'dislocations hand', 'house fire', 'Pneumoretroperitoneum', 'Blow -accident', 'chemical accident', 'cut hands', 'burns cold', 'burns corneal', 'Radiocapitellar dislocation', 'shoulder sprained', 'spinal fractures', 'abrasions vaginal', 'ecstasy overdose']
    

## 9. Mental or Behavioral Dysfunction


```python
print(list(df[df['STY'] == 'Mental or Behavioral Dysfunction']['STR'].sample(100)))
```

    ['exhibitionist', 'autistic disorders', 'Impaired memory', 'acute delirium', 'compulsive hoarding', 'delusions grandiose', 'Dysprosody', 'Aphasic children', 'substance use', 'Ejaculatio Praecox', 'night wakes', 'secondary dementia', 'Aphasia', 'Paranoid Schizophrenia', 'disorder mood', 'Nicotine Addiction', 'derealization', 'addictions exercise', 'amnestic syndrome', 'Alcoholic paranoia', 'sexual frustration', 'acute psychosis', 'Catatonic excitation', 'manic behavior', 'Post-pregnancy depression', 'Communicative disorders', 'occupational neurosis', 'Opium abuse', 'Stubbornness', 'conversion aphonia', 'Dissociative Disorder', 'personality change', 'Sexual Assault', 'Secondary Acalculia', 'coprophagy', 'hysteria mass', 'opiate addiction', 'quarrelsomeness', 'Self-injurious behaviors', 'Language disorders', 'mental block', 'Gustatory Agnosia', 'hemispatial neglect', 'middle insomnia', 'teeth grinding', 'Dissociative amnesia', 'Nervous tremulousness', 'Sleep terror', 'Sleep disorder', 'rape-trauma syndrome', 'cheek bite', 'caffeine withdrawal', 'gynephobia', 'Anxiety disorders', 'disability intellectual', 'Somatosensory agnosia', 'Acquired Dysgraphia', 'Childhood psychosis', 'Tongue biting', 'psychogenic hyperventilation', 'Behavioral abnormalities', 'Frigidity', 'Drunk driving', 'Anxiety Disorders', 'family dysfunction', "munchausen's syndrome", 'amnesia dissociative', 'absent mind', 'Delayed Orgasm', 'patient abuse', 'Adjustment disorders', 'Withdrawal Syndrome', 'behavior psychotic', 'frotteurism', 'night terrors', 'abuse steroid', 'Psychogenic fatigue', 'Communication impairment', 'emancipation disorder', 'hysterical disorder', 'night wake', 'Mental deterioration', 'schizotypal disorder', 'pathological gambling', 'Temper Tantrums', 'Hysterical Personality', 'heavy drinker', 'Mental Deterioration', 'Inferiority complex', "Broca's Aphasia", 'Postpartum Depression', 'Amnesic aphasia', 'Behavior problems', 'Developmental Agnosia', 'rumination disorder', 'Anepia', 'Spousal Abuse', 'Secondary Traumatization', 'Neuroticism', 'Ideomotor Apraxia']
    

## 10. Neoplastic Process


```python
print(list(df[df['STY'] == 'Neoplastic Process']['STR'].sample(100)))
```

    ['Systemic Mastocytosis', 'neoplasms', 'Ovarian Endometrioma', 'Oral leukoplasia', 'Extragonadal Seminoma', 'Avian Leukosis', 'Angiocentric Glioma', 'adenoma lung', 'tumor mouth', 'Tumorlet', 'Lymphoepithelial carcinoma', 'Neuroma', 'Nasal Neoplasms', 'Tumor', 'Colorectal Neoplasms', 'inflammatory breast cancer', 'Recurrent Rhabdomyosarcoma', 'Apocrine hidrocystoma', 'Cerebellar glioma', 'Odontogenic Myxofibroma', 'Cervix Adenocarcinoma', 'Angiogenic Switch', 'chondroid syringoma', 'systemic mastocytosis', 'mandible tumor', 'pecoma', 'Iris Melanoma', 'serous cystoma', 'plasmacytoma', 'androblastomas', 'myelomonocytic leukemia', 'mediastinum tumor', 'Ovarian Lymphoma', 'hand lipoma', 'Vulvar Trichoblastoma', 'Resectable Cholangiocarcinoma', 'Epiglottic Cancer', 'Therapy-Associated Neoplasms', 'colon carcinoma', 'Vagina Cancer', 'Osteoclastoma', 'oncolysis', 'stromal tumour', 'leukemia lymphoblastic', 'Ureteral Schwannoma', 'Chicken Hepatoma', 'Pituitary somatotropinoma', 'Pericardial Mesothelioma', 'Endometrial Neoplasms', 'adrenal cancer', 'Adult Ependymoblastoma', 'cancer eye', 'Tonsil Tumor', 'Metaplastic Thymoma', 'Adenoma sebaceum', 'Other Malignancy', 'Myocardial Neoplasm', 'bone cancer', 'Eruptive Collagenoma', 'carcinoid appendix', 'Bilateral Carcinoma', 'Thymic Neoplasia', 'buttock tumor', 'ovary tumour', 'Facial Neoplasms', 'heart neoplasms', 'parosteal osteosarcoma', 'muscle cancers', 'Papillary Cystadenoma', 'Adenocystic Carcinoma', 'lymphangioma', 'skin polyps', 'leukoplakias', 'Shope Papilloma', 'viral leukemogenesis', 'anus tumor', 'Lung Meningioma', 'neck tumor', 'Familial Glomangioma', 'testicular lymphoma', 'skin metastasis', 'Endodermal Rest', 'Feline Melanoma', 'tumor vulva', 'cystic lymphangioma', 'Gonadotropinoma', 'Laryngeal carcinoma', 'gingival tumor', 'bone tumor', 'urological cancer', 'penile neoplasms', 'hepatocellular adenomas', 'Mouse Myeloproliferation', 'Malignant Adenofibroma', 'Eyelid Tumor', 'Hobnail Hemangioma', 'Signet-Ring Melanoma', 'neuroendocrine carcinoma', 'pulmonary metastasis', 'Nevus sebaceus']
    

## 11. Pathologic Function


```python
print(list(df[df['STY'] == 'Pathologic Function']['STR'].sample(100)))
```

    ['venous stenosis', 'fetal death', 'obstetrical complications', 'Agalactorrhea', 'moist gangrene', 'allergy caffeine', 'fetal distress', 'pathogeneses', 'Testicular Hemorrhage', 'Ulcer', 'hydropic degeneration', 'Fat Embolism', 'Intracerebral Hemorrhage', 'complete miscarriage', 'eyelid oedema', 'Pseudarthrosis', 'cerebral edema', 'abscess gums', 'allergic shock', 'complications pregnancies', 'heterotopias', 'steatonecrosis', 'Anal Hemorrhage', 'Pulmonary Edema', 'Heterotopias', 'rh incompatibility', 'nasal boil', 'effects late', 'Anaphylactic Shock', 'pregnancy triplet', 'Maceration', 'Hemorrhagic Stroke', 'Cerebral ischemia', 'pregnancy hemorrhoids', 'Eosinophilic Granuloma', 'chronic obstruction', 'hemosiderin pigmentation', 'cardiac complications', 'chronic abscess', 'cold abscess', 'neoplasm invasiveness', 'post-dates', 'Posttransfusion purpura', 'cervical lesion', 'cancer complication', 'Intestinal bleeding', 'oedema legs', 'whiteheads', 'Epidermal thickening', 'lead line', 'problem menstrual', 'Infectious Granuloma', 'Friction rub', 'incomplete abortion', 'posttraumatic headache', 'Enlarging Abdomen', 'aneurysm leak', 'occluding', 'Food allergies', 'Post-Traumatic Headache', 'placenta abruptio', 'Hyphemia', 'Thromboembolic events', 'Cervical Insufficiency', 'complications haemodialysis', 'bacterial overgrowth', 'amyotrophies', 'Unequivocal Progression', 'fluid overload', 'shock syndrome', 'artery occlusion', 'photosensitivity', 'dysplasia', 'Hemorragia intermenstrual', 'Metrorrhagia', 'premenopausal menorrhagia', 'embolus arterial', 'Arterial Embolism', 'uterus hemorrhage', 'pregnancy hemorrhage', 'Cerebral Hemorrhage', 'Fat Atrophy', 'dystrophy', 'effect increased', 'haemorrhage conjunctiva', 'Choroidal Neovascularization', 'incompetence', 'potato allergy', 'crepitus', 'bleeding periods', 'laminar necrosis', 'complications injuries', 'sacral oedema', 'hemorrhages mucosal', 'buttocks hematoma', 'Subconjunctival haemorrhage', 'physiological shock', 'Dyserythropoiesis', 'Atypical Hyperplasia', 'cerebral hemorrhage']
    

## 12. Sign or Symptom


```python
print(list(df[df['STY'] == 'Sign or Symptom']['STR'].sample(100)))
```

    ['foot pronation', 'nail splitting', 'enanthema', 'Abdominal tenderness', 'tenting skin', 'Aprosodia', 'shoulder pain', 'pain rest', 'Soif excessive', 'tummy ache', 'burps', 'joint tenderness', 'changes stools', 'stool medication', 'mouth sloughing', 'Decorticate Rigidity', 'Dizzy', 'Depression aggravated', 'eye twitching', 'spurling sign', 'burn urination', 'Pelvic Pain', 'Cushingoid', 'Akoria', 'Muscle weakness', 'hand tremor', 'Decorticate State', 'pain ankle', 'elbows pain', 'Mittelschmerz', 'itching scalp', 'dry eyes', 'arm pain', 'Finger stiffness', 'myoclonic jerks', 'lips numbness', 'Cerea Flexibilitas', 'Fishy odor', 'Belch', 'aching', 'night sweat', 'blushing', 'telescoping', 'mental exhaustion', 'Pronated Foot', 'muscle cramp', 'pain heart', 'Watery eyes', 'mental exhaustion', 'Constipation', 'Rood oog', 'fremitus', 'retractions supraclavicular', 'pain scrotal', 'Paracusis', 'jitteriness', 'Dilated Anus', 'neurological disability', 'Sensitive lips', 'synkinesias', 'coughing exercise', 'Hallucinations Auditory', 'Neonatal cyanosis', 'gum irritations', 'back pain', 'nipples tenderness', 'hearts pain', 'grunting respiration', 'sensations tingling', 'breasts sores', 'Dolor ocular', 'itching hands', 'abdominals bloated', 'Nasal regurgitation', 'ischialgia', 'eye ache', 'generalized pruritus', 'Flasher', 'nauseating', 'aching limbs', 'pain sit', 'flaking nails', 'cluttering', 'Muscular spasticity', 'numbness throat', 'foot pronation', 'Headache Frontal', 'vomiting medication', 'sitting pain', 'Limb rigidity', 'Papulopustular Rash', 'generalized pain', 'arms weakness', 'hyperacusia', 'myoclonic disorder', 'aggravated depression', 'chest burning', 'hot flushes', 'Feeling Cold', 'Convulsive Seizures']
    

# Chemical and drugs

1. Amino Acid, Peptide, or Protein
2. Antibiotic
3. Biologically Active Substance
4. Biomedical or Dental Material
5. Chemical
6. Chemical Viewed Functionally
7. Chemical Viewed Structurally
8. Clinical Drug
9. Element, Ion, or Isotope
10. Enzyme
11. Hazardous or Poisonous Substance
12. Hormone
13. Immunologic Factor
14. Indicator, Reagent, or Diagnostic Aid
15. Inorganic Chemical
16. Nucleic Acid, Nucleoside, or Nucleotide
17. Organic Chemical
18. Pharmacologic Substance
19. Receptor
20. Vitamin

## 1. Amino Acid, Peptide, or Protein


```python
print(list(df[df['STY'] == 'Amino Acid, Peptide, or Protein']['STR'].sample(100)))
```

    ['propionyl-vitellogenin', 'Incivek', 'Hb Watts', 'Mucolator', 'Importin', 'Phosphorylase ab', 'Phosvitin', 'adiponitrile amidase', 'Hb F-Sacromonte', 'elastase inhibitor', 'Haptoglobin-Related Protein', 'idazoxan receptors', 'L-asparagine', 'Infergen', 'gamma-hydroxybutyrate receptor', 'lamins', 'neuropeptide', 'Sperm Receptors', 'Tankyrases', 'Aurora-A Kinase', 'diaphorase', 'Exteins', 'Multi-domain protein', 'polyphosphatase', 'N-acetylglycine', 'deoxycholate hydroxylase', 'Phosphatidylserine Synthase', 'diaminoacid', 'lysin', 'anti ige', 'Attractin', 'lactoferroxin A', 'liver enzyme', 'guanase', 'Linear Gramicidin', 'glucagon recombinant', 'glucosidosucrase', 'Gentiobiase', 'peptide tyrosine-tyrosine', 'Rapicidin', 'Argonaute', 'Somatropin recombinant', 'S-Type Lectins', 'amine oxidase', 'Pregabalin', 'deoxyribonuclease I', 'tissue prokallikrein', 'Decapeptyl', 'Microsomal Monooxygenase', 'Stefin-A', 'abarelix', 'Hexarelin', 'Hb Bucuresti', 'Component', 'varicella-zoster immunoglobulin', 'Takhzyro', 'Sucly', 'swivelase', 'Synapsin I', 'Methyl aspartate', 'polymyxin e', 'depsipeptide', 'Monoclonal antibodies', 'Hb Yakima', 'iletin', 'zinc hydroaspartate', 'Nck-Interacting Kinase', 'peptide C', 'Hb Bruxelles', 'Hb Wuming', 'trauma peptide', 'Cergutuzumab Amunaleukin', 'Laminin', 'nucleoside phosphorylase', 'tetrahydrobiopterin', 'Hyaluronan Receptor', 'Bradykinin Receptors', 'Hb J-Buda', 'Cytokine', 'Glaucescin', 'Endostatin', 'cubulin', 'Prolactin Receptors', 'Peanut Lectin', 'alpha-methylalanine', 'tgf beta', 'Hemoglobin Osler', 'Apolipoproteins C', 'Hb Bibba', 'mannan-binding protein', 'Chimeric Protein', 'Aggregoserpentin', 'denileukin diftitox', 'recombinant somatotropin', 'Plastid Proteins', 'penicilamina', 'Biphasic Insulins', 'tumor-specific antigen', 'Phospholipid Scramblase', 'ribosome receptors']
    

## 2. Antibiotic


```python
print(list(df[df['STY'] == 'Antibiotic']['STR'].sample(100)))
```

    ['ceftiofur sodium', 'Besifloxacin Hydrochloride', 'Cefadroxil Anhydrous', 'Mithramycinum', 'Cubicin', 'pibrozelesin', 'Heliomycin', 'Anti-Bacterial Compounds', 'Claramid', 'Pivampicillin Hydrochloride', 'Relomycin', 'Ticillin', 'Amrubicin', 'Vancomycine', 'Aknemycin', 'aminopenicillin', 'mikamycin A', 'Negamicin', 'Oxytetracycline', 'Geopen', 'Rokitamycin', 'Actinomycin C', 'vantin', 'antibacterial drug', 'bactroban', 'c mitomycin', 'Quinolone Antimicrobial', 'Triadcortyl', 'Propikacin', 'polymyxin B', 'Neomycin', 'Crystallinic Acid', 'Amikin', 'Dihydrostreptomycin Sulfate', 'Actinomycin C', 'Septacef', 'Sirolimús', 'Quartermaster', 'Dibekacin', 'Oxitetraciclina', 'desquinolone', 'Cefurox', 'sulphadiazine', 'targocid', 'cephalosporin antibiotics', 'roxithromycin', 'Acanya', 'nebcin', 'rubidazone', 'Cleeravue-M', 'Bio-Mycin', 'Apalcillin', 'Benaxima', 'ascomycin', 'cephaloridin', 'Moxifloxacin Hydrochloride', 'Apacef', 'Hetacilina', 'Cytovaricin', 'Caprolactam', 'trimethoprim-sulfadoxine', 'Sulfametazyny', 'Aknosan', 'Cefadroxil monohydrate', 'aminoglycosides antibiotics', 'pen-vee', 'Griseofulvin microsize', 'Roxithromycinum', 'Geocillin', 'Cardinophillin', 'ácido fusídico', 'capreomycin', 'Sissomicin', 'Amphotec', 'Factive', 'Az-threonam', 'hitachimycin', 'Arbekacinum', 'Cefetecol', 'Kempi', 'Doxorubicin hydrochloride', 'teicoplaninum', 'Oxytétracycline', 'antibiotic combinations', 'antibiotics skin', 'Viomycin', 'Josamycin', 'Temocillin', 'Astromicin Sulfate', 'macrolide antibiotic', 'sulphadiazine', 'Neo-Decadron', 'adoxa', 'coli mycin', 'Razupenem', 'Cefadyl', 'Ceftriaxon', 'Myciguent', 'paromomycin', 'sisomicin']
    

## 3. Biologically Active Substance


```python
print(list(df[df['STY'] == 'Biologically Active Substance']['STR'].sample(100)))
```

    ['acid malic', 'salivatin', 'Isomaltose', 'Pregnancy Proteins', 'Somatomedin A', 'Lithocholyltaurine', 'cyclo-psychotride A', 'Prostatein', 'Avenin', 'Alpha-Catenin', 'hemoglobin Quin-Hai', 'Hb Kempsey', 'Clerodane', 'Apolipoprotein D', 'Xylulose', 'hemophil', 'cysteines', 'endozepines', 'Protein Sck', 'bcl-Xalpha Protein', 'hemoglobin C', 'coenzymes', 'Caffeic Acids', 'christmas factor', 'Plastoquinone', 'Acido chenodeoxicholico', 'Glycerol Phosphoglycerides', 'Surface Glycoproteins', 'frataxin', 'a-dna', 'Protohemin', 'Chimeric Protein', 'Dystrophin-Related Protein', 'Bile pigments', 'Argonaute', 'Hexaectylic acid', 'Acute-Phase Reactants', 'oxldl', 'Lecithin', 'Guanidine Monohydrate', 'Midkine', 'Ovomucoid', 'bodies ketones', 'Sialoproteins', 'Thyroxine-Binding Prealbumin', 'betulonolic aldehyde', 'microorganism toxin', 'Colostral-Val nonapeptide', 'Hb Johnstown', 'Plasma Protein', 'Metmyoglobin', 'Leukotriene C', 'Ferredoxin I', 'antisense oligonucleotides', 'Pheromone', 'Sterigmatocystin', 'Stigmasterol', 'irbic acid', 'urea nitrogen', 'neutrophin', 'Hemoglobin Potomac', 'Transcriptional Coactivator', 'Placental Proteins', 'Enteramine', 'Thymidin', 'chimeric protein', 'Pregnanolone', 'biologic agent', 'antithrombin I', 'beta-Sialoglycoprotein', 'bursopentine', 'Hb Saverne', 'Adjusted calcium', 'Inositide Phospholipids', 'hemoglobin Maputo', 'Bacterial Proteins', 'Chenodiol', 'Dystroglycans', 'auxin', 'alpha-solamarine', 'Iron-sulfur proteins', 'Thymine', 'Neuraminic acid', 'pidolic acid', 'procalcitonin', 'tumor markers', 'c-myb Proteins', 'Phospholipid Scramblase', 'chylomicron', 'Beriplex P-N', 'biondianoside F', 'Hb Titusville', 'Elk-L Protein', 'lysobisphosphatidic acid', 'myeloma globulin', 'Biologic Medicines', 'dentin phosphoprotein', 'Microsatellites', 'melibiose permease', 'heat-shock protein']
    

## 4. Biomedical or Dental Material


```python
print(list(df[df['STY'] == 'Biomedical or Dental Material']['STR'].sample(100)))
```

    ['Implast', 'surgi primebond', 'Deodorants', 'Saliva Natura', 'methyl methacrylate', 'Feeder Cell', 'Supramid', 'Soft Cap', 'Ionos cement', 'Tablet Triturate', 'Topical Gel', 'Ophthalmic Solutions', 'Polawax Polysorbate', 'pharmaceutical preservatives', 'vinylformic acid', 'Cutaneous paste', 'fl optibond', 'Dentifrice Powder', 'glass-carbomer', 'Soluble tablet', 'dental resin', 'vehicle', 'creams topical', 'sublingual spray', 'beta-Dextrin', 'Dental gel', 'Intrauterine System', 'Coated Tablet', 'polyvinyl', 'ointments topical', 'microfilm', 'coating tablets', 'ferrofluid', 'Rectal Suppositories', 'microcrystalline celluose', 'Magnetic-Targeted Carriers', 'alginate calcium', 'Intramammary solution', 'liquid bandage', 'Monodehydrosorbitol Monooleate', 'Duprene', 'Enteric-coated tablets', 'Sodium Cocoamphoacetate', 'Implantable Pellet', 'Tissue scaffolds', 'Scotchcast', 'Soft Capsule', 'oral suspension', 'Nasal Ointment', 'Silochrome', 'extract', 'beta-whitlockite', 'Vehicle', 'impression material', 'Sublingual Tablet', 'ophthalmic suspension', 'nylon', 'Vaginal Suppository', 'Micro Enema', 'jelly', 'for Solution', 'Unmedicated Sponge', 'Plasdone', 'Bone Cements', 'irrigation solution', 'one step', 'infusion powder', 'Liquid bandage', 'polyvinyl chloride', 'Isononyl isononanoate', 'Buccal film', 'Octylphenoxy Polyethoxyethanol', 'Ceram-X', 'Spherex', 'Drug Implant', 'biodegradable polymer', 'beta-whitlockite', 'beta-Cyclodextrin', 'ocrylate', 'Polyvinyl alcohol', 'Scotchcast', 'Biotrey cement', 'Fluoristat', 'Porcelain', 'Suspending Agent', 'Intrauterine emulsion', 'Graft material', 'irrigating solutions', 'Hybrid Composite', 'Tincture', 'coated drugs', 'cholesteryl isostearate', 'pudding', 'wetting agents', 'formocresol', 'Allograft', 'prevocel', 'topical spray', 'Biocompatible Materials', 'Surfactant']
    

## 5. Chemical


```python
print(list(df[df['STY'] == 'Chemical']['STR'].sample()))
```

    ['Compound']
    

## 6. Chemical Viewed Functionally


```python
print(list(df[df['STY'] == 'Chemical Viewed Functionally']['STR'].sample(100)))
```

    ['odorant', 'Flame Retardants', 'pharmaceutic aids', 'Chemical Probe', 'Detergents', 'Differentiation Agents', 'degreasers', 'laboratory chemicals', 'Aerosol propellants', 'antineoplastons', 'Organic Oxide', 'Esperamicins', 'pro-drugs', 'flavor enhancers', 'drugs pro', 'Disodium Eosin', 'adjuvants vaccine', 'indigo carmine', 'ingredient', 'Eosin Y', 'Pharmaceutical Adjuvant', 'pigment', 'flavor additive', 'Molecular Target', 'Gadolinium-Chelant Complex', 'flavos', 'neurotransmitter inhibitor', 'Sequestering Agent', 'antineoplaston', 'flavoured', 'pro drug', 'Adjunct Agent', 'substitute sugar', 'agents volatile', 'Pharmaceutical Adjuvants', 'Pyrophosphoric Acids', 'Eosin Y', 'Flavoring', 'Fire Retardants', 'inhalants volatile', 'Hair Colorants', 'essential nutrients', 'flavoring', 'Nutritive Sweeteners', 'Esperamycin', 'Oxidants', 'Molecular Target', 'flavored', 'flavour', 'steroid cream', 'chemical preservatives', 'sugar substitutes', 'biocoating', 'Esperamycin', 'food color', 'azofuchsine', 'Adjuvant', 'Chemical Modifier', 'Caustic substance', 'Differentiating Agents', 'Whitening Agents', 'chemical preservatives', 'melanin inhibitor', 'Detergent', 'Embolic Bead', 'Oxidants', 'quinoline yellow', 'adjunct agent', 'essential nutrients', 'chemical laboratory', 'Bleaching Agents', 'propellant', 'sweetener', 'bulk-forming agent', 'Explosives', 'Drug Precursors', 'synthetic surfactant', 'Eosine G', 'tartrazine', 'machining fluid', 'Oxytocin Blocker', 'Cardiovascular Agent', 'substitute sugars', 'Gadolinium-Chelate', 'Flavoring', 'Decorporation Agent', 'Chemical Probe', 'Organic Impurities', 'Odorant', 'Bulk-forming Agent', 'Lewis Bases', 'Differentiation Inducer', 'Eosin Yellowish', 'Indigotin', 'Static eliminators', 'Non-Nutritive Sweeteners', 'pharmaceutic aids', 'steroid cream', 'Esperamicin', 'Eosine']
    

## 7. Chemical Viewed Structurally


```python
print(list(df[df['STY'] == 'Chemical Viewed Structurally']['STR'].sample(100)))
```

    ['tautomer', 'Fullerene', 'sulphides', 'copolymer', 'arsenical', 'gravels', 'chemical fumes', 'free radical', 'Lead compounds', 'Ferric Compounds', 'Theaflavin', 'sulfur compounds', 'gas fume', 'Iron compounds', 'compound sulphur', 'Buckminsterfullerenes', 'Nickel Compound', 'carbon nanotubes', 'disulfide', 'Nickel compounds', 'fullerenes', 'Phosphines', 'Liposome Vesicle', 'hypophosphites', 'Emulsion', 'Gravel substance', 'Lead compounds', 'hot liquids', 'sulfur compounds', 'arsenical', 'Macromolecular Compounds', 'Ammonium Compounds', 'Indium compounds', 'analog', 'Molecular Target', 'Salts', 'Liquid Crystals', 'chemical fumes', 'solution uw', 'Arsenic Compound', 'borohydride', 'Dermal Filler', 'Coordination Complexes', 'Divalent Cations', 'cations', 'Heteropolymer', 'Ketac cem', 'Nitrites', 'dimer', 'Chemical Modification', 'hydroxy compound', 'Rhenium compounds', 'Polymeric', 'Polymeric Macromolecules', 'Nobudyne', 'hypophosphite', 'radical', 'Polymorphic Crystals', 'solution uw', 'Dendrons', 'Phosphonous Acids', 'Dermal Fillers', 'Polymer', 'gravels', 'sulfur acid', 'phosphonous acid', 'salts', 'Nitrates', 'gases', 'arsenic compounds', 'dioctyldimethylammonium chloride', 'Structural Modifier', 'Vapor', 'Crystal Structure', 'Phosphorus compounds', 'Hypophosphorous Acids', 'enantiomers', 'Gases', 'uw solution', 'Complexes', 'vapors', 'buckyballs', 'Thiosulfonic Acids', 'Phosphoranes', 'emulsion', 'zinc compounds', 'Belzer solution', 'solid state', 'borates', 'crystals', 'azide', 'sulfur acids', 'dimer', 'crystal', 'Polymers', 'Macromolecules', 'Boron compounds', 'Macromolecular Branch', 'Nanoparticle Functionality', 'Chemical Modification']
    

## 8. Clinical Drug


```python
print(list(df[df['STY'] == 'Clinical Drug']['STR'].sample(100)))
```

    ['rynatan tablet', 'water solution', 'Magic Mouthwash', 'Doans Pill', 'tretinoin topical', 'sodium oral', 'gentamicin ophthalmic', 'topical hydrocortisone', 'Nicotine Inhalant', 'urea topical', 'corticosteroids creams', 'permethrin lotion', 'diazepam injection', 'corticosteroid cream', 'Sinelee Patch', 'epinephrine cisplatin', 'dimethicone cream', 'sesal yeongo', 'cipro tablets', 'testosterone gel', 'Carmustine Wafer', 'correctol', 'sinecatechins ointment', 'cipro tablets', 'budesonide oral', 'Busulfan Injection', 'cream efudix', 'inderal tablets', 'oral budesonide', 'progesterone topical', 'nicotine patch', 'testosterone patch', 'Nicotine Patch', 'Betadine topical', 'lozenges nicotine', 'oral miconazole', 'natalizumab Injection', 'Nicotine Patch', 'Nicotine Gum', 'ustekinumab Injection', 'monolayers', 'monolayer', 'canakinumab Injection', 'hydron', 'carmustine implant', 'povidone-iodine solution', 'Viokase Powder', 'furosemide injection', 'nystatin topical', 'cream metronidazole', 'hydrocortisone topical', 'gel isotrex', 'eyes gentamicin', 'neomycin topicals', 'injections procrit', 'donnatal tablets', 'topical ibuprofen', 'sulfur topical', 'senna tabs', 'dimenhydrinate injection', 'bc powders', 'topical epinephrine', 'pill sleeping', 'fluorouracil topical', 'amobarbital injection', 'glycerin enema', 'hydrocortisone ointments', 'topical lidocaine', 'levoleucovorin Injection', 'onabotulinumtoxinA 200 UNT Injection [Botox]', 'Flavonoid Tablet', 'lacticare lotion', 'diflucan tablet', 'Sinelee Patch', 'topical ketoconazole', 'phenergan tablet', 'betadine douche', 'progesterone topical', 'topical epinephrine', 'cough syrups', 'lopressor tablets', 'Urisal Tablet', 'Neo-Synalar Cream', 'Testosterone Gel', 'correctol', 'povidone-iodine solution', 'ziconotide Injection', 'granules perdiem', 'lorazepam injection', 'injection pfizerpen', 'Nicotine Inhalant', 'hydrocortisone topical', 'topical tacrolimus', 'gels lubricating', 'senna tabs', 'guaifenesin syrup', 'fluocinonide cream', 'glycerin suppositories', 'biaxin filmtab', 'topical calcitriol']
    

## 9. Element, Ion, or Isotope


```python
print(list(df[df['STY'] == 'Element, Ion, or Isotope']['STR'].sample(100)))
```

    ['selenium', 'selenate ion', 'Ferric Cation', 'gas element', 'Gd element', 'Eco-Plus', 'Dietary Silicon', 'yttrium sesquioxide', 'Os element', 'Potassium cation', 'arsenic', 'Halogens', 'Osmium Metallicum', 'iridium', 'Oxygen', 'rhenium', 'Ar element', 'cadmium cation', 'ytterbium', 'Tetrachloroaurate ion', 'plutonium', 'Radioactive Iodine', 'uranium trioxide', 'Della Soft', 'liquid nitrogen', 'Charcodote', 'protium', 'magnesium ions', 'cobaltous cation', 'Tin Radioisotopes', 'potassium', 'antimony', 'Mo element', 'nickel', 'Tl element', 'active oxygen', 'Bicarbonate Ions', 'Oxygen Radioisotopes', 'Potassium metal', 'Sodium Radioisotopes', 'Ytterbium', 'black carbon', 'Graphene', 'Arsenite ion', 'Deuterium ions', 'Gallium Isotopes', 'halogens', 'Actinides', 'Pb element', 'columbio', 'Phosphate Ion', 'yttrium', 'Uranium', 'Plutonium', 'Elemental calcium', 'Uuh ununhexium', 'potassium ions', 'Europium oxide', 'titanous hydroxide', 'Nitrogen Isotopes', 'Active Oxygen', 'scandium', 'gallium', 'free calcium', 'Sodium cation', 'Sodium Ion', 'Barium Radioisotopes', 'Krypton', 'Barium Ion', 'gold drug', 'Heavy ions', 'molybdenum', 'Mc moscovium', 'Charcoal', 'Ba++ element', 'na sodium', 'Neptunium', 'Erbium Metallicum', 'samarium', 'Ipratropium ion', 'monofluorophosphate ion', 'Iodip', 'Liquid nitrogen', 'strontium cation', 'Rubidium Radioisotopes', 'Na+ element', 'Ca++ element', 'Nitrogen Isotopes', 'dichromate ion', 'cobalt', 'Della Care', 'Sr++ element', 'Adsorba', 'graphite', 'Oxonium ions', 'Calcium', 'Dietary Selenium', 'Chlorine', 'Yttrium Radioisotopes', 'anions']
    

## 10. Enzyme


```python
print(list(df[df['STY'] == 'Enzyme']['STR'].sample(100)))
```

    ['thiocyanate hydrolase', 'L-Tryptophan Aminotransferase', 'aspartate N-acetyltransferase', 'quinolinate phosphoribosyltransferase', 'Ficin', 'A deoxyribonuclease', 'Acid Beta-Galactosidase', 'hydroxylamine oxidase', 'dihydrodihydroxybenzoate dehydrogenase', 'selenocysteine methyltransferase', 'inosinicase', 'Aag enzyme', 'Chymotrypsin B', 'Endoglycosidase D', 'alpha-glucosidase', 'triterpenol esterase', 'tyrosine-arginine synthetase', 'Pyrimidine-nucleoside phosphorylase', 'phosphotriose isomerase', 'phosphomonoesterase', 'ecto-atpase', 'Capsaicin-Hydrolyzing Enzyme', 'Cathepsin O', 'polynucleotide ligase', 'quinolinate phosphoribosyltransferase', 'Thyroid Galactosyltransferase', 'retinal photoisomerase', 'ck bb', 'G proteins', 'Carboxypeptidase Q', 'phosphoglycerate dehydratase', 'Neprilysin', 'Monoglyceride Lipases', 'Manganese Catalase', 'cardiolipin synthase', 'tilactase', 'Novo Alcalase', 'diisopropyl fluorophosphatase', 'triose-phosphate isomerase', 'Lipoate Acetyltransferase', 'Polyphosphate kinase', 'glycoprotein palmitoyltransferase', 'glycosidase', 'Xanthine Oxidoreductase', 'Sfericase', 'Anionic Trypsinogen', 'o-pyrocatechuate decarboxylase', 'deoxyhypusine monooxygenase', 'protein farnesyltransferase', 'fructose biphosphatase', 'atpase', 'Uridine-Cytidine Kinase', 'Gq Protein', 'Ptprz Phosphatase', 'Acrosin', 'tubulin-tyrosine ligase', 'kynurenine aminotransferase', 'protein geranylgeranyltransferase', 'Acid Maltase', 'Pantothenate kinase', 'Erwinia L-asparginase', 'luciferin sulfokinase', 'A Transferase', 'Triacylglycerol acylhydrolase', 'Cyclo-Oxygenase', 'sulphatase', 'furin', 'Exo-Cellobiohydrolase', 'D-mannuronate lyase', 'erythrocyte cholinesterase', 'Phenolsulfotransferase P', 'acylpeptide hydrolase', 'Peptidoglycan N-acetylmuramoylhydrolase', 'aspartyl proteases', 'Lactose Galactohydrolase', 'dynamin', 'Lipoprotein Lipase', 'Spermine Synthase', 'Ascorbase', 'dehydrogenase ketoglutarate', 'homocysteine synthase', 'Serine Palmitoyltransferase', 'trihydroxynaphthalene reductase', 'Brachyurin', 'nickel-iron hydrogenase', 'Heme-Controlled Inhibitor', 'Phosphotransacylase', 'Sucrase-Isomaltase Complex', 'preproacrosin', 'rci recombinase', 'cyclopropane synthetase', 'Cholenzyme', 'Retinol-Palmitate Synthetase', 'Myrosinase', 'Proline Dehydrogenase', 'Carbon-Nitrogen Lyases', 'Onconase', 'Aminohydrolases', 'Macrophage-Specific Metalloelastase', 'N-Acetyllactosamine Synthetase']
    

## 11. Hazardous or Poisonous Substance


```python
print(list(df[df['STY'] == 'Hazardous or Poisonous Substance']['STR'].sample(100)))
```

    ['ammonium chloroplatinate', 'Snuff', 'botulin', 'Difolatan', 'Oxyparathion', 'Vinylbenzene', 'Ronnel', 'Crotalotoxin', 'acephate', 'Acetophenetidin', 'crocidolite', 'organochlorine insecticide', 'mekarzole', 'bacterial enterotoxins', 'dichloran', 'lead carbonate', 'nitrafen', 'butachlor', 'vinylidene fluoride', 'Tetraethyl Lead', 'carbon nitride', 'Animal toxin', 'Monocrotaline', 'Aminopterin', 'shigella toxin', 'Sépou', 'Pain killers', 'hazardous chemicals', 'Toad Venom', 'dimethoxystrychnine', 'renal toxin', 'Benzidine', 'Nobecutan', 'dicloran', 'Parazene', 'Ethyl Mesylate', 'butox', 'repellent', 'phenylmercury', 'Soman', 'drugs speed', 'Botulin', 'magic mushrooms', 'alpha-Chloroacetophenone', 'Toxohormone', 'parasite carcinogen', 'Methyldimethylaminoazobenzene', 'Karbofos', 'trinitrotoluene', 'Methyl Alcohol', 'azinphos methyl', 'Scorpion Venom', 'Chlorofluorocarbons', 'chlorohydrocarbon insecticide', 'phyton', 'Clorfenvinfos', 'monosodium methanearsonate', 'Het Acid', 'Vinyl Chloride', 'trinitrotoluene', 'Crotylaldehyde', 'fenvalarate', 'opiate alkaloid', 'N-nitrosohexamethyleneimine', 'war gas', 'Dietilestilbestrol', 'hexyl cinnamylaldehyde', 'venom coagglutinin', 'Toluene Diisocyanate', 'trichlorofluoromethane', 'Alkylating Agents', 'Hexachlorocyclohexane', 'Methamphetaminum', 'carbon disulphide', 'Formaldehyde solution', 'Bioallethrin', 'pyrenes', 'Araneid Venom', 'Thiophanate-Methyl', 'Unden', 'aldoxycarb', 'Exfoliatins', 'crack cocaine', 'Carbonic dichloride', 'glufosinate-ammonium', 'tralopyril', 'dimetan', 'Mercury', 'Trifluralin', 'dimethylnitrosomorpholine', 'Nickelous Sulfate', 'calomel', 'Clonitralide', 'Defibrase', 'O-Dianisidine', 'staphylolysin', 'disodium methanearsonate', 'Vulklor', 'Carcinogen', 'Organofluorine compounds']
    

## 12. Hormone


```python
print(list(df[df['STY'] == 'Hormone']['STR'].sample(100)))
```

    ['Pergestron', 'dinoprosta', 'gut hormones', 'Abaloparatide', 'serum cortisol', 'Other hormones', 'fempatch', 'neurohormones', 'neotenin', 'plant hormone', 'Solu-Medrone', 'Thymalin', 'L-noradrenaline', 'Thyroid Antagonists', 'Nandrolone phenylpionate', 'Lutropin alfa', 'Zumenon', 'natriuretic hormone', 'Androstanedione', 'Etiocholanolone Glucuronide', 'Progestational Agents', 'Beclometasone', 'Octrotide', 'Neumune', 'deaminodicarbaoxytocin', 'androderm', 'ortho-cept', 'Caberdelta M', 'Fluoximesteron', 'Neo Fertinorm', 'Fludroxycortide', 'lente insulin', 'Lutrepulse', 'metipregnone', 'Cortifoam', 'Angiotensinamide', 'Nu-Megestrol', 'Thyroid antagonists', 'cervidil', 'Quinestradol', 'Lysine Vasopressin', 'Eston', 'Humulin R', 'thyroid-stimulating hormone', 'Bétaméthasone', 'Overstin', 'Liraglutide', 'free testosterone', 'Secretin human', 'Thymosin Fs', 'beta-Methasone alcohol', 'Nutropin', 'Pregneninolone', 'Colprosterone', 'Somatomedins', 'boldenone', 'terlipressin', 'Gestonorone Caproate', 'Anabolic Steroids', 'androsterone', 'Ghrelin', 'pralmorelin', 'Calcitriol-Nefro', 'Cryo-Tropin', 'Micronephrine', 'Thyrotropic-releasing factor', 'diethylstilbestrol propionate', 'crinone', 'Cosyntropin', 'Insect Hormones', 'Recombinant Gonadotropin', 'ethinylestradiol', 'Leios', 'Somatrogon', 'segesterone acetate', 'Prednilem', 'Humulin R', 'Pamorelin', 'deoxycortone pivalate', 'gamma-Lipotropin', 'delatestryl', 'alpha subunit', 'Prolactin', 'norethisterone acetate', 'Vasopressin', 'Calcitonin Salmon', 'Fertirelin Acetate', 'analogs hormone', 'Recombinant Inhibins', 'activin B', 'Osteotriol', 'nortestosterone phenylpropionate', 'velosulin', 'Recombinant Glucagon', 'Taltirelin', 'Somatomedin', 'metenolone', 'triptoreline', 'Epoetinum zeta', 'Terlipresina']
    

## 13. Immunologic Factor


```python
print(list(df[df['STY'] == 'Immunologic Factor']['STR'].sample(100)))
```

    ['infliximab-dyyb', 'Rotavirus antigen', 'Simtuzumab', 'Gamma globulins', 'bacterial immunostimulant', 'Iveegam', 'antigen f', 'Fowlpox Vaccine', 'antigens jk', 'Hickory antigen', 'Cemiplimab-rwlc', 'hcv anti', 'Lupus Anticoagulant', 'Viral antigens', 'demcizumab', 'Enfortumab Vedotin', 'lymphokine', 'Hypervariable Loop', 'Tetanus immunoglobulin', 'Lymphotoxin-alpha', 'dia antibody', 'snake antivenom', 'Immunoconjugate', 'hilda', 'cyclophilin D', 'Fareletuzumab', 'Bacterial Capsule', 'Etrolizumab', 'Naptumomab Estafenatox', 'Tagraxofusp', 'belimumab', 'tissue factor', 'm antibody', 'Intramuscular immunoglobulin', 'plantain antigen', 'synthetic vaccine', 'Immunoglobulin G', 'Chimeric Immunoreceptors', 'Anavip', 'Dematiaceae antigen', 'Cétuximab', 'Sacituzumab Govitecan', 'Nanobodies', 'Sabin Vaccine', 'Avicine', 'broncho-vaxom', 'Vepalimomab', 'Urokinase Receptor', 'Sialophorin', 'Glomulin', 'Sargramostatin', 'Aldesleukina', 'Rabbit antigen', 'Gammar-P', 'H-D Antigens', 'anticoagulants lupus', 'isoantibodies', 'Engerix-B', 'rimabotulinumtoxin B', 'antibody h', 'antinuclear factors', 'Afutuzumab', 'Begelomab', 'Pork Antigen', 'agglutinogen', 'aging insulin', 'Wegener Autoantigen', 'gamimune', 'Enoticumab', 'Galactoglycoprotein', 'superantigens', 'cemiplimab', 'Prostatac', 'Staphylolysin antigen', 'Antiphospholipid Antibody', 'immunoglobulins lambda', 'Prostate-Specific Antigen', 'Fimbrial Adhesins', 'chemokine', 'Bacterial Adhesin', 'mmr vaccines', 'Synthetic antigens', 'histone ab', 'Mls Determinants', 'house dust', 'Defensins', 'Pm bacterin-toxoid', 'factor P', 'lapuleucel-T', 'Ocrevus', 'Vantictumab', 'receptor antibody', 'pembrolizumab', 'Adalimumab', 'Ulocuplumab', 'iga secretory', 'immunosuppressants', 'Polyvalent Vaccine', 'Rubella Vaccine', 'Tigatuzumab']
    

## 14. Indicator, Reagent, or Diagnostic Aid


```python
print(list(df[df['STY'] == 'Indicator, Reagent, or Diagnostic Aid']['STR'].sample(100)))
```

    ['amaranth', 'Bac-Seph', 'zinc iodide-osmium', 'Phloroglucin', 'polymetaphosphate', 'Lauroyl sarcosine', 'Eluent', 'alpha-isonitropropiophenone', 'tolonium chloride', 'Tetramethylphenylenediamine', 'selenophenol', 'norstatine', 'sodium cyanide', 'paraphenylene-diamine', 'para-caprylamidophenol', 'Bromine Cyanide', 'contrast material', 'azofloxin', 'm-aminobenzeneboronic acid', 'Tartrazine Barium', 'mercuric dibromide', 'chromium dichloride', 'p-azidophenobarbital', 'Lymphazurin', 'chelating iron', 'tuberculin', 'zinc iodide-osmium', 'Lauroyl Sarcosine', 'Benzylpenicilloyl-polylysine', 'cellulose polyphosphate', 'Ethaneperoxoic acid', 'contrast dyes', 'radiopharmaceutical', 'spin label', 'Safranine T', 'agents diagnostic', 'Antisense Oligonucleotides', 'methylenium ceruleum', 'Feridex', 'stains', 'chromogens', 'Meglumine Diatrizoate', 'phosphorus thiotrichloride', 'brilliant green', 'deferoxamine', 'gastrographin', 'ortho-Phthalic Aldehyde', 'guaiac', 'Dimyristoyllecithin', 'carbol fuchsin', 'Desiccants', 'phytic acid', 'beta-Aminopropionitrile', 'Radio-Hippuran', 'Niagara Blue', 'Acetyl Hydroperoxide', 'styrofoam', 'Methylchloroform', 'dibenzosuberone', 'sodium triphosphate', 'Phenylthiourea', 'Miniruby', 'Methylphenazonium Methosulfate', 'Helicosol', 'horseradish peroxidase', 'Bunamiodyl', 'Ioxitalamic acid', 'Urotrast', 'gadolinium', 'Calcium Iopodate', 'Iodamide Meglumine', 'Polyaziridine', 'Coprin', 'Minims Stains', 'cumylhydroperoxide', 'monoferric edetate', 'Oncoscint Prostate', 'Iodotope', 'Dicyclohexylcarbodiimide', 'Trypan Red', 'ortho-fluorophenylalanine', 'radioactive drug', 'Biligrafine', 'cortrosyn', 'Peroxidase stain', 'tachypyridine', 'brazilin', 'p-Aminohippuric Acid', 'Amaranth Dye', 'ferrocyanide', 'Reno M-Dip', 'Bromocresol Purple', 'Embedding Agent', 'Rechtsweinsäure', 'Phthalic Anhydrides', 'zirconyl phosphate', 'light green', 'Antisense Agents', 'dimethoxystrychnine', 'gadofosveset']
    

## 15. Inorganic Chemical


```python
print(list(df[df['STY'] == 'Inorganic Chemical']['STR'].sample(100)))
```

    ['phosphorus white', 'electrolyte', 'Chromium alloys', 'Alum Compounds', 'Feridex', 'Hydrogen fluoride', 'hopeite', 'S-Hydril', 'cinnabar', 'Hydrogen Peroxide', 'phosphonic acid', 'bismuth carbonate', 'alcides', 'durasphere', 'Dental Amalgams', 'Silica gel', 'tetraamine platinum', 'Diatomaceous silica', 'Mercury compounds', 'azarcon', 'Bum Ease', 'Free Radioiodine', 'arsenic sulfide', 'selenite-cisplatin conjugate', 'phosphate calcium', 'tricalcium orthophosphate', 'Flura-Loz', 'Neutracare', 'Magnesium Chloride', 'Mineral Fiber', 'Yieronia', 'ringer solution', 'Water Ice', 'Calcium Oxide', 'mercury vapors', 'halides', 'Cupric Ion', 'opoka dust', 'dialose', 'Salinaax', 'Iron pentacarbonyl', 'aluminium chloride', 'Neoscan', 'salt water', 'Muriatic Acid', 'Altamist', 'minerals water', 'Gastro-Hek', "Lassar's Paste", 'Iodide', 'cadmium sulphide', 'phosphorus trihydroxide', 'Cis-platinum', 'Hydroxyl', 'Gallium compounds', 'Hagan Phosphate', 'Aldrox', 'Blue Vicking', 'Pertechnetate Sodium', 'Fluorodex', 'ferric pyrophosphate', 'Transactinide Elements', 'Tiomolibdate Diammonium', 'Orthoboric Acid', 'sodium borohydride', 'nitrogen oxide', 'iron carbonyl', 'Ammonium Molybdate', 'lead sulfide', 'diarsenic trioxide', 'silica gel', 'potassium tetrahydroborate', 'hexametaphosphate', 'beryllofluoride', 'Cobalt-Chromium Alloys', 'magnesium iodide', 'Vanadium Compounds', 'Selenos', 'selenite', 'Haematite', 'gold ash', 'vitapex', 'chromic nitrate', 'Cyanogen bromide', 'monomethyl phosphate', 'Sodium Biselenite', 'Calvital', 'stannic oxide', 'Nanocis', 'Benedikt reagent', 'Potassium supplement', 'Duraflor', 'Dialuminum trisulfate', 'tetraarsenic oxide', 'sodium dithionate', "lugol's iodine", 'Calcium Carbimide', 'electrolytes potassium', 'Red Phosphorus', 'Peroxynitrous Acid']
    

## 16. Nucleic Acid, Nucleoside, or Nucleotide


```python
print(list(df[df['STY'] == 'Nucleic Acid, Nucleoside, or Nucleotide']['STR'].sample(100)))
```

    ['adenazole', 'Guanosine Diphosphomannose', 'Floxuridine', 'cytidylic acid', 'Telbivudina', 'Triflorothymidine', 'Cytosar-U', 'adenosine diphosphate', 'Adenylylsulfate', 'Major Groove', 'E-Box Motifs', 'Zoliparin', 'iododeoxyuridine', 'trifluridine', 'L-Phenylisopropyladenosine', 'Apy compound', 'Trifluridin', 'Affinitak', 'Leustatine', 'Inosine Monophosphate', 'isobarbituridine', 'Uridine Diphosphate', 'Futraful', 'Virophta', 'repeats tandem', 'Pyrimidine Antagonist', 'adenosine monophosphate', 'halopurine', 'guanosine triphosphate-sepharose', 'Ribovirin', 'uridine diphosphate', 'ganciclovir', 'Deazauridine', 'Aciclobeta', 'Adenosine phosphate', 'chloroethylguanine', 'Poly C', 'Q base', 'Gemcitabine-Oxaliplatin Regimen', 'Adenylpyrophosphate', 'Trifluorothymine deoxyriboside', 'deoxythymidine triphosphate', 'poly adenine', 'mtdna', 'Replication-Competent Retroviruses', 'Guanosine Pyrophosphate', 'Fluorodeoxyuridine', 'viramidine', 'Cidofovir', 'E-Box Elements', 'Oxiamin', 'Aciclostad', 'Oblimersen Sodium', 'trecs', 'denavir', 'Delta Elements', 'Deoxyribozymes', 'Propylthiouracilum', 'Ropidoxuridine', 'nucleoside sulfate', 'Fluorocyclopentenylcytosine', 'triphosphopyridine nucleotide', 'Guanidine derivatives', 'minisatellite repeats', 'uridyladenosine', 'zidovudine triphosphate', 'viramidine', 'Triacetyl uridine', 'Combivir', 'cytidine diphosphate', 'Imunovir', 'Fluorodeoxyuridylate', 'Adenylylimidodiphosphate', 'succinylzidovudine', 'Floxuridin', 'Futraful', 'Poly I-C', 'Cordycepin Triphosphate', 'Adenine Nucleotides', 'methisoprinol', 'Aciclovir-Sanorania', 'Replication-Competent Retroviruses', 'cytarabine hydrochloride', 'cidofovir', 'Adenine Arabinoside', 'thymine riboside', 'Liposomal cytarabine', 'Brivudine Phosphoramidate', 'Antisense Oligoribonucleotides', 'Neofluor', 'Decitabine', 'benzimidavir', 'Adenosine', 'Aracytidine', 'Puromycine', 'Aclovir', 'Cyclobutane-Pyrimidine Dimers', 'guadecitabine', 'Guanosine Pentaphosphate', 'deoxyribonucleoside']
    

## 17. Organic Chemical


```python
print(list(df[df['STY'] == 'Organic Chemical']['STR'].sample(100)))
```

    ['azithromycin anhydrous', 'Quifenadine', 'pirbuterol acetate', 'ortho-Aminobenzoic Acids', 'Bromotheophylline', 'Mannan', 'Aloin', 'Chlordecone', 'Apo-Oxybutynin', 'Meprobamato', 'Indoramine', 'cardiotonic steroids', 'Homapin', 'Cyproterone acetate', 'dovonex', 'Unilax', 'fradiomycin sulfate', 'Apressoline', 'oxyethyltheophylline', 'chenodeoxycholic acid', 'Myristyl Alcohol', 'Fluprednisolone Valerate', 'Hexopal', 'cholesterol', 'Bardoxolone methyl', 'Bethanidine', 'mepiphylline', 'Stugeron Forte', 'Pirenzepine', 'norplants', 'Salmeterol Xinafoate', 'para-nitrophenylphosphorylcholine', 'p-methoxybenzenediazoniumfluoroborate', 'Iodobenzoates', 'Calcium Aminosalicylate', 'kavinton', 'virkon', 'asulacrine', 'Mercazolyl', 'Sospitan', 'Rupatadine', 'Dical-D', 'Mexitil', 'Mendi', 'disaturated phosphatidylcholine', 'Gastrozepin', 'conjugated oestrogens', 'Enterodez', 'alpha-humulene', 'baeckeside', 'Fluspiperone', 'enoxolone dipotassium', 'ergotrate', 'Alpheprol', 'Flexeril', 'Impromidine Trihydrochloride', 'bipinnatin-B', 'sulfophosphatidylcholine', 'Fungazoil', 'pulmicort respule', 'ethylbenzoylecgonine', 'trioxifene methanesulfonate', 'Absorbine', 'Ketorolaco', 'Furacin', 'acetyl carnitine', 'decamethonium iodide', 'Heparin Sodium', 'Jacobaea Maritima', 'Dimethylchondrocurarine', 'Aureolic Acid', 'benadryl allergy', 'Mebezonium Iodide', 'ixazomib citrate', 'ginkgolide-J', 'Dimemorfan', 'para-azbenzenetrimethylammonium', 'Benzydamine Monohydrochloride', 'ethyl heptoate', 'Minodronic Acid', 'centellicin', 'Theocolin', 'Hydroxyethylstarch', 'inmercarb', 'psolusoside-B', 'Spermaceti', 'bricanyl', 'Aknichthol', 'Rhotrimine', 'Phosphotrienin', 'Lagodeoxycholic Acid', 'Letrozol', 'Docosyl alcohol', 'naringin sodium', 'sodium levulinate', 'polyether urethane', 'estramustine phosphate', 'Radilem', 'porphin', 'iodochlorhydroxyquin']
    

## 18. Pharmacologic Substance


```python
print(list(df[df['STY'] == 'Pharmacologic Substance']['STR'].sample(100)))
```

    ['Pyrazineamide', 'Spirodiflamine', 'caldiamide sodium', 'Ispaghula husk', 'mercuhydrin', 'Nitramyl', 'gluconic acid', 'nitrodur', 'Exohemagglutinins', 'Magnesium gluconicum', 'amphetamine sulfate', 'dehydrocacalohastin', 'Polividone', 'Clomidazole', 'Carbatuss Syrup', 'ritodrine', 'aggrastat', 'Urovist', 'Viscosupplements', 'Yeast-X External', 'Bromerguride', 'Gallium nitrate', 'Tronolane Suppositories', 'Predfoam', 'Dopalina', 'Antibody-Toxin Hybrids', 'dimebone', 'Gapicomine', 'Capval', 'oticaine', 'oxametacine', 'Imipramine', 'Riboflavin', 'drugs fertility', 'lauryl mercryl', 'Bismutite Hemihydrate', 'Lithium chloride', 'methylacetylenic putrescine', 'Nuelin', 'dokloxythepine', 'Bactifor', 'Cinnarizine', 'Synamol', 'methylarecaidin', 'Apo-Sulin', 'Neuractil', 'Cyclomenol', 'Secretagogues', 'Panamesine', 'N-desmethylchlorpromazine', 'epinephryl borate', 'Atysmal', 'Travel-Ease', 'dimethothiazine', 'acetaminophen-codeine', 'Edaravone', 'Hexamarium Bromide', 'Chiggerex', 'Irtemazole', 'acetyl carnitine', 'Prunetol', 'Phosphonoformic Acid', 'acetaminophen pentazocine', 'Ladakamycin', 'Norimipramine', 'chenodeoxycholic acid', 'angel trumpet', 'dichloroisocyanuric acid', 'antiviral', 'Nesvacumab', 'Xyrem', 'cardilate', 'difezil', 'Amphenicols', 'nifluril', 'Curanail', 'antimalarial agents', 'Trifosmin', 'Lefamulin', 'Levora', 'Cardiomax', 'Varespladib Methyl', 'Enciprazine Dihydrochloride', 'pantothenol', 'Ethamsylate', 'Heparinic acid', 'Propel', 'conjugate vaccines', 'Vicrom', 'curatoderm', 'ethamivan', 'Proroxan', 'neutrexin', 'Cystatin A', 'maruyama vaccine', 'Plasmasteril', 'Soothe Caplets', 'Vardenafil Hydrochloride', 'Hydrocortisonum', 'clorothepin']
    

## 19. Receptor


```python
print(list(df[df['STY'] == 'Receptor']['STR'].sample(100)))
```

    ['adrenoreceptor', 'Laminin Receptors', 'Receptors', 'Fractalkine', 'Fas-Like Protein', 'Contactin-Associated Protein', 'Narcotic Receptors', 'Notch Proteins', 'Surface Immunoglobulins', 'Corticotropin Receptors', 'charybdotoxin receptors', 'c-mas protein', 'Neurophilin', 'Calcium Receptors', 'thymulin receptor', 'Notch Receptors', 'Neuropeptide Receptors', 'Notch Protein', 'hemopexin-heme receptor', 'Abiotic Receptors', 'Torpedo syntrophin', 'testosterone receptors', 'Fas Antigen', 'Laminin Receptor', 'Imidazoline Receptors', 'reactive site', 'nephritogenic glycopeptide', 'Cholecystokinin Receptor', 'Purine Receptor', 'Integrins', 'nociceptin receptors', 'tsh receptor', 'purinergic receptors', 'metallothionein receptors', 'sigma Receptors', 'Phencylidine Receptor', 'sulfatide receptors', 'Sulfonylurea Receptors', 'beta-glycan', 'Opioid Receptors', 'ldl receptor', 'Cyclophilin A', 'Glutamate Receptors', 'Asialoorosomucoid Receptor', 'beta glycan', 'vanilloid receptors', 'nuclear receptor', 'Rosette Receptor', 'Ubi-L receptors', 'Opioid Receptors', 'Gastrin Receptor', 'Olfactory Receptor', 'hydrocortisone-receptor complex', 'Stanolone Receptor', 'ferritin receptors', 'Chemokine Receptor', 'transcobalamin receptors', 'Neurotrophin Receptors', 'Transferrin Receptor', 'Glucagon-Like Receptors', 'Muscle-Specific Kinase', 'Smoothened Homolog', 'vasopressin receptors', 'Neuromedin-B Receptor', 'lipophorin receptors', 'binding sites', 'Phencyclidine Receptors', 'estriol receptors', 'ach receptors', 'Muscle Caveolin', 'oxysterol receptor', 'scavenger receptors', 'scyllatoxin receptors', 'Glycine Receptor', 'Virus Receptors', 'sigma Receptor', 'chylomicron receptors', 'Glucocorticoid Receptor', 'ingenoid receptor', 'adrenoreceptor', 'Enkephalin Receptors', 'Vasopressin Receptors', 'lipolysis-stimulated receptors', 'Thrombopoietin Receptors', 'Diazepam Receptors', 'receptors tnf', 'Calcium-Sensing Receptor', 'Toll-like receptors', 'Cholecalciferol Receptor', 'Sortilin-Related Receptor', 'fsh receptor', 'Calcitonin Receptor', 'ferritin receptors', 'adhesion receptors', 'Purinergic Receptor', 'Histamine receptor', 'ferriexochelin receptors', 'B-Chemokine Receptor', 'receptor cannabinoid', 'dihydromorphine receptors']
    

## 20. Vitamin


```python
print(list(df[df['STY'] == 'Vitamin']['STR'].sample(100)))
```

    ['methyl folate', 'lutein ocuvite', 'Mephyton', 'Vita-E', 'Retinyl Aldehyde', 'alphacalcidol', 'Menatetrenone', 'neurobions', 'Premesis Rx', 'lipoic acid', 'folinic acid', 'Pantothenate Calcium', 'Tocolion', 'tiamina', 'Menaquinone', 'Cocarboxylase', 'Nicomide', 'Elex Verla', 'alpha-tocopherol hydroquinone', 'Detulin', 'Hidroferol', 'Rodex', 'Cobavite', 'Epit Vit', 'Levocarnitine Acetyl', 'palmitate ascorbyl', 'rodex', 'Ecalcidene', 'Phyllochinonum', 'Retinaldehyde', 'Ascorbyl Palmitate', 'Ferotrinsic', 'Hydroxy-Cobal', 'Tocopheryl acetate', 'thiaminium', 'Alitretinoin', 'Hydroxocobalamine', 'fat-soluble vitamin', 'acids lipoic', 'Chiro-Inositol', 'Cenolate', 'Methyltocols', 'Paroven', 'maternity', 'Multivitamin preparation', 'Nikotinsäure', 'alpha e', 'pyridoxine hcl', 'Menadiol diphosphate', 'Multi-Delyn', 'Vita-E succinate', 'calcium pantothenate', 'Calcium Leucovorin', 'alpha-tocopheronolactone', 'benfotiamina', 'Leukovorin', 'dibenzoylthiamine', 'Pantothenate', 'Provitamins', 'Bioflavonoids', 'Hybrin', 'p-Carboxyaniline', 'trigonellinamide', 'vitamin hair', 'Calcamine', 'methyl folate', 'Vitamin', 'Beta-cryptoxanthin', 'Hydroxocobalamin Acetate', 'thiamine nitrate', 'chromagen-ob', 'Coenzyme R', 'tocopherols', 'Folamin', 'doxercalciferol', 'Methylcobalamin', 'Lard Factor', 'Acide folinique', 'Riboflavin Mononucleotide', 'glutofac zx', 'Biovit-A', 'berocca', 'beta-Butoxyethyl nicotinate', 'Aminobenzoic Acids', 'niacin', 'ergocalciferol', 'Lithium Nicotinate', 'Ferotrinsic', 'inecalcitol', 'Phylloquinone', 'Pantenyl', 'Cyanokit', 'nephro-vite', 'Cobanamida', 'certa-vite', 'Avitol', 'Biopto-E', 'Acerola', 'D-pantothenic acid', 'vicon forte']
    


```python

```


```python

```
