import openai
import pandas as pd
import json

# Load your API key
openai.api_key = 'your-api-key-here'

# Load the DataFrame
data = pd.read_csv('path_to_your_csv_file.csv')  # make sure your CSV has 'problem' and 'rationale' columns

def process_rationale(problem, rationale):
    """Use GPT-3.5 to split the rationale into reasoning steps."""
    prompt = f"You are asked to finish a data augmentation job. Specifically, you will given a problem that can be solved by Chain-Of-Thought reasoning, and you will be given a rationale that displays some preliminary hints to the problem. Your job is to split the rationale into multiple reasoning steps. Specifically, output a JSON formatted file with entry pairs of 'current reasoning step', 'reasoning result' until the problem is solved. <Problem>: {problem}, <rationale>: {rationale}"
    response = openai.Completion.create(
        model="gpt-3.5-turbo",  # or another appropriate model version
        prompt=prompt,
        max_tokens=500,  # Adjust based on the length needed
        temperature=0.5  # Adjust for creativity level; lower is more deterministic
    )
    return response.choices[0].text.strip()

# Process each row in the DataFrame and store results
results = []
for index, row in data.iterrows():
    result = process_rationale(row['problem'], row['rationale'])
    results.append(json.loads(result))  # Assuming the response is already in JSON format

# Save the results to a JSON file
with open('output.json', 'w') as f:
    json.dump(results, f)

print("Data processing complete, output saved to output.json")
