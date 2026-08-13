from datasets import load_dataset
import json

# Load dataset
dataset = load_dataset('MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT')

# Save to local file
with open('data/final_training_dataset.jsonl', 'w') as f:
    for item in dataset['train']:
        json.dump(item, f)
        f.write('\n')

print('Dataset downloaded successfully!')