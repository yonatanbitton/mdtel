import os

# module_path = os.path.abspath(os.path.join(os.getcwd()))
# print(f"In config, appending {module_path}")
# sys.path.append(module_path)

data_dir = r"E:\mdtel_data\data" + os.sep
umls_similarity_path = r'E:\umls_perl\UMLS-Similarity-1.47\utils'
FINAL_LABELS_COL = 'merged_inner_and_outer'

disorder_lines = """
DISO|Disorders|T020|Acquired Abnormality
DISO|Disorders|T190|Anatomical Abnormality
DISO|Disorders|T049|Cell or Molecular Dysfunction
DISO|Disorders|T019|Congenital Abnormality
DISO|Disorders|T047|Disease or Syndrome
DISO|Disorders|T050|Experimental Model of Disease
DISO|Disorders|T033|Finding
DISO|Disorders|T037|Injury or Poisoning
DISO|Disorders|T048|Mental or Behavioral Dysfunction
DISO|Disorders|T191|Neoplastic Process
DISO|Disorders|T046|Pathologic Function
DISO|Disorders|T184|Sign or Symptom
"""

chemical_lines = """
CHEM|Chemicals & Drugs|T116|Amino Acid, Peptide, or Protein
CHEM|Chemicals & Drugs|T195|Antibiotic
CHEM|Chemicals & Drugs|T123|Biologically Active Substance
CHEM|Chemicals & Drugs|T122|Biomedical or Dental Material
CHEM|Chemicals & Drugs|T103|Chemical
CHEM|Chemicals & Drugs|T120|Chemical Viewed Functionally
CHEM|Chemicals & Drugs|T104|Chemical Viewed Structurally
CHEM|Chemicals & Drugs|T200|Clinical Drug
CHEM|Chemicals & Drugs|T196|Element, Ion, or Isotope
CHEM|Chemicals & Drugs|T126|Enzyme
CHEM|Chemicals & Drugs|T131|Hazardous or Poisonous Substance
CHEM|Chemicals & Drugs|T125|Hormone
CHEM|Chemicals & Drugs|T129|Immunologic Factor
CHEM|Chemicals & Drugs|T130|Indicator, Reagent, or Diagnostic Aid
CHEM|Chemicals & Drugs|T197|Inorganic Chemical
CHEM|Chemicals & Drugs|T114|Nucleic Acid, Nucleoside, or Nucleotide
CHEM|Chemicals & Drugs|T109|Organic Chemical
CHEM|Chemicals & Drugs|T121|Pharmacologic Substance
CHEM|Chemicals & Drugs|T192|Receptor
CHEM|Chemicals & Drugs|T127|Vitamin
"""

disorder_tuis = [l.split("|")[2] for l in disorder_lines.split("\n") if l != '']
chemical_or_drug_tuis = [l.split("|")[2] for l in chemical_lines.split("\n") if l != '']

DISORDER = "Disorder"
CHEMICAL_OR_DRUG = "Chemical or drug"

DISORDERS_COL = "Disorders"
CHEMICAL_OR_DRUGS_COL = "Chemical or drugs"

SIMILARITY_THRESHOLD = 0.88

general_exceptions = ['מחלה', 'תרופה']

LOW_SIMILARITY_THRESHOLD = 0.80

# High recall matcher
LOW_SINGLE_WORD_SIMILARITY_THRESHOLD = 0.78
UP_SINGLE_WORD_SIMILARITY_THRESHOLD = 0.85
LOW_MULTI_WORD_SIMILARITY_THRESHOLD = 0.83
UP_MULTI_WORD_SIMILARITY_THRESHOLD = 0.90
DIABETES, SCLEROSIS, DEPRESSION = 'diabetes', 'sclerosis', 'depression'
NUMBER_OF_GRAMS = 3
STRING_COLUMN = 'STR'
number_of_posts_needed = {'sclerosis': 259, 'diabetes': 259, 'depression': 262}
