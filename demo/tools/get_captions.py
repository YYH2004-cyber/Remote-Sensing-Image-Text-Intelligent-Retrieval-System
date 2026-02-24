import json
import pprint

with open('/home/dhm/python_project/demo/tools/dataset_rsicd.json', 'r') as f:
    data = json.load(f)
    # print(data.keys())
    # print(data['images'][0].keys())
    pprint.pprint(data['images'][0]['sentences'])
    # ... existing code ...
    with open('/home/dhm/python_project/demo/tools/captions.txt', 'w') as f:
        seen_sentences = set()  # Track unique sentences
        total_sentences = 0
        unique_sentences = 0
        for i in data['images']:
            for j in i['sentences']:
                total_sentences += 1
                sentence = j['raw']
                if sentence not in seen_sentences:  # Only write if not already seen
                    seen_sentences.add(sentence)
                    f.write(sentence + '\n')
                    unique_sentences += 1
        
        print(f"Total sentences processed: {total_sentences}")
        print(f"Unique sentences written: {unique_sentences}")
        print(f"Duplicates removed: {total_sentences - unique_sentences}")
# ... existing code ...