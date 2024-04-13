import json
import sys
import os

def getch():
    # Assuming Unix-like system for the example
    import tty
    import termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def parse_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['outputs']

def verify_predictions(data):
    correct_indices = []
    index = 0

    while index < len(data):
        item = data[index]
        while True:
            print(f"\nEvaluating {index + 1}/{len(data)}")
            print("Prediction: ", item['prediction'])
            print("Target: ", item['target'])
            print("Is the prediction correct? (q for correct, e for wrong, b to go back, \\ to exit): ")

            response = getch()
            if response.lower() == 'q':
                if index not in correct_indices:
                    correct_indices.append(index)
                break
            elif response.lower() == 'e':
                if index in correct_indices:
                    correct_indices.remove(index)
                break
            elif response.lower() == 'b':
                if index > 0:
                    index -= 2  # Reduce by 2 because the increment at the end of the loop will add back one
                break
            elif response.lower() == '\\':
                print("Exiting evaluation.")
                sys.exit(0)
            else:
                print("Invalid input. Only 'q', 'e', 'b', or '\\' are accepted. Please try again.")
        index += 1

    return correct_indices.sort(), len(data)

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
    output_file = file_path.rsplit('.', 1)[0] + '_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")

def main():
    try:
        if len(sys.argv) != 2:
            print("Usage: python script.py <path_to_directory>")
            sys.exit(1)
        
        folder_path = sys.argv[1]
        if not os.path.isdir(folder_path):
            print("Error: The path provided is not a directory")
            sys.exit(1)

        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
        if not json_files:
            print("No JSON files found in the directory.")
            return

        print(f"{len(json_files)} JSON file(s) found:")
        for json_file in json_files:
            file_path = os.path.join(folder_path, json_file)
            print(f"Processing {json_file}...")
            data = parse_json(file_path)
            correct_indices, total_count = verify_predictions(data)
            accuracy = calculate_accuracy(correct_indices, total_count)
            
            print(f"\nIndices marked as correct: {correct_indices}")
            print(f"Total Correct: {len(correct_indices)} out of {total_count}")
            print(f"Accuracy: {accuracy:.2f}")
            
            save_results(file_path, correct_indices, total_count, accuracy)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting gracefully.")
        sys.exit(0)

if __name__ == "__main__":
    main()
