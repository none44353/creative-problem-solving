import sys
import collections 

# 此处解决因Python版本过高而造成的collections引用错误的问题
if sys.version_info.major == 3 and sys.version_info.minor >= 10:
    from collections.abc import MutableSet, MutableMapping
    collections.MutableSet = collections.abc.MutableSet
    collections.MutableMapping = collections.abc.MutableMapping

import os
import json
import openai
import logging

from copy import deepcopy
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


def save_to_file(path, data, filename):
    """
    Save data to a JSONL file.

    Args:
    - path: The directory path for the file.
    - data: The data to save, as a dictionary.
    - filename: Name of the JSONL file.
    """
    file_path = os.path.join(path, filename)
    with open(file_path, "a") as f:
        f.write(json.dumps(data) + "\n")


def query(path, data, logger, client, timestamp=None):
    """
    Handles parallel API requests to OpenAI with retry logic for each request.

    Args:
    - path: String for log tracking, representing the request context.
    - data: List of dictionaries, each containing 'model', 'messages'.

    Returns:
    - List of processed responses, maintaining the original input order.
    """
    # Generate timestamped folder path for saving files
    if timestamp == None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path = f"tmp/{path}/{timestamp}"
    os.makedirs(folder_path, exist_ok=True)

    # Save all input data to input.jsonl before any requests are made
    for item in data:
        save_to_file(folder_path, {key: item[key] for key in item if key != "process_function"}, "input.jsonl")

    responses = [None] * len(data)  # List to store responses in the correct order
    with ThreadPoolExecutor() as executor:
        futures = []
        for idx, item in enumerate(data):
            future = executor.submit(_query_with_retries, idx, item, logger, client)  # Submit each query with retry support
            futures.append(future)

        # Collect and organize completed responses in the correct order
        for future in as_completed(futures):
            results = future.result()
            responses[results[0]] = results

    # Save all processed output data to output.jsonl after all requests are completed
    for idx, response_data in enumerate(responses):
        if response_data:
            _, response, _ = response_data
            save_to_file(folder_path, response, "output.jsonl")

    return [processed_answer for _, _, processed_answer in responses]


def _query_with_retries(idx, data, logger, client):
    """
    Executes API requests with retry logic for resilient querying.

    Args:
    - idx: Index of the query in the data list.
    - data: Dictionary representing a query including model, messages, etc.
    
    Returns:
    - List of tuples (index, combined_response, processed_answers) for each successfully processed response.
    """
    # Set maximum retries based on data["n"]
    max_retries = int(data["n"] * 2 + 5)

    # Prepare the process function and initialize the response container
    process_function = data.get("process_function", 
        lambda response: response["choices"][0]["message"]["content"])  # Default processing function
    api_data = {key: value for key, value in data.items() if key != "process_function"}

    # Prepare the response and processed answers
    combined_response = None
    processed_answers = []
    n = data.get("n", 1)

    for attempt in range(max_retries):
        # Perform the query
        response = _query_openai(api_data, logger, client)
        if response is not None:
            try:
                # Append each choice to combined_response["choices"]
                if combined_response is None:
                    combined_response = response
                else:
                    combined_response["choices"].extend(response["choices"])
                
                # Process each answer and add to processed_answers list
                # print(response)
                processed_answer = process_function(response)
                processed_answers.append(processed_answer)
                
                # Break if we have gathered enough responses
                if len(processed_answers) >= n:
                    processed_answers = processed_answers[:n]  # Limit to n answers if extra gathered
                    break
            except Exception as e:
                m = len(processed_answers)
                if attempt >= max_retries - 1:
                    logger.error(f"Error processing response on attempt with reties {max_retries - n} for index {idx}: {e}")
                # logger.error(f"Error processing response on attempt {attempt - m + 1}/{max_retries - n + m} for index {idx}: {e}")
        else:
            # m = len(processed_answers)
            if attempt >= max_retries - 1:
                logger.error(f"Error processing response on attempt with reties {max_retries - n} for index {idx}")
            # logger.error(f"API call failed on attempt {attempt - m + 1}/{max_retries - n + m} for index {idx}")

    return idx, combined_response, processed_answers


def _query_openai(data, logger, client):
    """
    Sends a single request to the OpenAI API.

    Args:
    - data: Dictionary containing the 'model' and 'messages' for the request.

    Returns:
    - API response as a dictionary if successful; None if an error occurs.
    """
    try:
        response = client.chat.completions.create(**data)
        return response.to_dict()  # Convert response to dictionary format
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        return None


def test(questions, logprobs=False, top_logprobs=None, temperature_list=None):
    """
    Test function to validate the 'query' function's handling of multiple requests, 
    ensuring responses match input order and expected answers.

    Verifies that responses:
    - Match the order of input questions.
    - Contain the expected answers for given test queries.
    """
    
    # 创建日志器
    logger = logging.getLogger("src/utils/query.py")
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # 创建格式化器并设置格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # 将处理器添加到日志器
    logger.addHandler(console_handler)

    # Initialize OpenAI client with OpenRouter base URL and API key from environment variables
    with open("utils/api_key.txt", "r") as file:
        api_key = file.read().strip()
    client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    
    test_data = [
        {
            "model": model,
            "messages": [{"role": "user", "content": question}],
            "process_function": lambda response: response["choices"][0]["message"]["content"],
            "n": 1,
            "logprobs": logprobs,
        }
        for question in questions
        for model in [
            "openai/gpt-4-turbo"
            # "openai/gpt-4o-mini-2024-07-18"
            # "openai/gpt-4o-2024-11-20"
            # "google/gemini-2.0-flash-001"
        ]
    ]
    if top_logprobs is not None:
        for data in test_data:
            data.update(top_logprobs=top_logprobs)
    if temperature_list is not None:
        new_test_data = []
        for data in test_data:
            for temperature in temperature_list:
                data_copy = deepcopy(data)
                data_copy.update(temperature=temperature)
                new_test_data.append(data_copy)
        del test_data
        test_data = new_test_data

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run query and retrieve responses
    query("test", test_data, logger, client, timestamp)
    
    return timestamp
