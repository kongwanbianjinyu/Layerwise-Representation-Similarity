import json

in_f = '/local/storage1/jiang.2880/SimCTG-main/document_generation/outputs/aligned_alternating_sampling_exit11.json'
with open(in_f) as f:
    item_list = json.load(f)

text_list = []
for item in item_list:
    text = item['generated_result']['0']['continuation']
    text_list.append(text)

# compute the evaluation results
from simctg.evaluation import measure_repetition_and_diversity
rep_2, rep_3, rep_4, diversity = measure_repetition_and_diversity(text_list)
print ('The result of rep-2 is {}, rep-3 is {}, rep-4 is {}, and diversity is {}'.format(rep_2, rep_3, rep_4, round(diversity,2)))
