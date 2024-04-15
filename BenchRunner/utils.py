import re


def extract_answer(input_string):
    """An automated parser to extract the answer from llm"""
    # Define the pattern to match the expression 'the answer is ...'
    pattern = r'the answer is (.*?)(?=\.)'

    # Search for the pattern in the input string
    matches = re.findall(pattern, input_string, re.IGNORECASE)

    # If a match is found, extract and return the answer
    if matches:
        answer = matches[-1]
        return answer

    return None
