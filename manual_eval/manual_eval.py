import json
import sys
## example usage: python3 manuel_eval.py boolean_expressions.json
## output: a .json file named "{input_file_name}_results.json"


def parse_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['outputs']

def verify_predictions(data):
    correct_indices = []
    total_count = 0
    
    for index, item in enumerate(data):
        while True:
            print("\nPrediction: ", item['prediction'])
            print("Target: ", item['target'])
            response = input("Is the prediction correct? (c for correct, w for wrong): ")
            
            if response.lower() == 'c':
                correct_indices.append(index)
                break
            elif response.lower() == 'w':
                break
            else:
                print("Invalid input. Only 'c' or 'w' are accepted. Please try again.")

        total_count += 1
    
    return correct_indices, total_count

def calculate_accuracy(correct_indices, total_count):
    correct_count = len(correct_indices)
    if total_count == 0:
        return 0  # Avoid division by zero
    return correct_count / total_count

def save_results(file_path, correct_indices, total_count, accuracy):
    results = {
        "correct_count": len(correct_indices),
        "total_count": total_count,
        "accuracy": accuracy,
        "correct_indices": correct_indices
    }
    output_file = file_path.rsplit('.', 1)[0] + '_results.json'  # Save as JSON with '_results' suffix
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_json_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    data = parse_json(file_path)
    correct_indices, total_count = verify_predictions(data)
    accuracy = calculate_accuracy(correct_indices, total_count)
    
    print(f"\nIndices marked as correct: {correct_indices}")
    print(f"Total Correct: {len(correct_indices)} out of {total_count}")
    print(f"Accuracy: {accuracy:.2f}")
    
    save_results(file_path, correct_indices, total_count, accuracy)

if __name__ == "__main__":
    main()

