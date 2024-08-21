import os
import sys
import time
import logging
from datetime import datetime

from anthropic import Anthropic
import pandas as pd
import regex as re

# User-defined constants
SYSTEM_PROMPT = 'You are a helpful assistant and a web development expert.'
CLAUDE_MODEL = "claude-3-sonnet-20240229"
MAX_TOKENS = 4096
PROMPT_INDEX_COLUMN = 'PROMPT_ID'  # Name of the column containing prompt indices
PROMPT_COLUMN = 'PROMPT'  # Name of the column containing the actual prompts
MAX_DELAY = 15  # Maximum delay between API calls in seconds

# Additional columns to include in results
ADDITIONAL_COLUMNS = ['TUTORIAL_PATH']

# Time-stamped output directory
TIME_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
DIR_RESULT = f"results_{TIME_STAMP}"
LogFileName = "LOG.log"
PKL_FILE_NAME = "RESULTS.pkl"

# Create directory to save results if it doesn't exist
os.makedirs(DIR_RESULT, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(os.path.join(DIR_RESULT, LogFileName)), logging.StreamHandler()])

def setup_anthropic_api():
    """Set up the Anthropic API key."""
    logging.info("Anthropic API key not found in environment variables. Please enter it now.")
    print("Anthropic API key not found in environment variables. Please enter it now.")
    api_key = input("Paste your Anthropic API key: ").strip()
    
    client = Anthropic(api_key=api_key)
    
    try:
        # There's no direct method to list models, so we'll make a simple API call to test authentication
        client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": "Hello"}]
        )
        logging.info("Anthropic API authentication successful.")
    except Exception as e:
        logging.error(f"An error occurred while setting up the Anthropic API: {str(e)}")
        sys.exit(1)
    
    return client

def generate_response(client, prompt_text, prompt_id):
    """Generate a response using Anthropic API based on user input."""
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=0.5,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": prompt_text
                }
            ]
        )
        prefix = str(prompt_id).zfill(4)
        result_content = response.content[0].text
        result_file_name = re.sub(r'\W+', '_', "result").lower()
        result_file_name = f"{prefix}_{result_file_name}.txt"
        with open(os.path.join(DIR_RESULT, result_file_name), "w", encoding="utf-8") as f:
            f.write(result_content)

        return result_content, response
    except Exception as e:
        logging.error(f"Error generating response: {str(e)}")
        return None, None

def process_prompts(client, path_to_prompts):
    """Process prompts and generate responses."""
    try:
        df = pd.read_pickle(path_to_prompts)
        logging.info(f"Columns in the dataframe: {list(df.columns)}")
        print(f"Columns in the dataframe: {list(df.columns)}")
        
        if PROMPT_COLUMN not in df.columns:
            raise KeyError(f"'{PROMPT_COLUMN}' column not found in the input DataFrame")
        
        if PROMPT_INDEX_COLUMN not in df.columns:
            raise KeyError(f"'{PROMPT_INDEX_COLUMN}' column not found in the input DataFrame")
        
    except FileNotFoundError:
        logging.error(f"Input file {path_to_prompts} not found.")
        return
    except KeyError as e:
        logging.error(f"Column error: {str(e)}")
        return
    except Exception as e:
        logging.error(f"Error reading input file: {str(e)}")
        return

    prompts_list = list(zip(df[PROMPT_INDEX_COLUMN], df[PROMPT_COLUMN]))
    
    results_list = []
    
    total_prompts_count = len(prompts_list)
    for index, (prompt_id, prompt_text) in enumerate(prompts_list, 1):
        # Check if this prompt has already been processed
        if any(item['PROMPT_ID'] == prompt_id for item in results_list):
            logging.info(f"Skipping prompt {index} (ID: {prompt_id}) as it's already processed.")
            continue

        logging.info(f"Processing prompt {index} of {total_prompts_count}")
        print(f"Processing prompt {index} of {total_prompts_count}")
        start = datetime.now()
        result_content, response = generate_response(client, prompt_text, prompt_id)
        if result_content:
            result_dict = {'PROMPT_ID': prompt_id, 'RESULT': result_content, 'RESPONSE': response}
            # Add additional columns
            for col in ADDITIONAL_COLUMNS:
                if col in df.columns:
                    result_dict[col] = df.loc[df[PROMPT_INDEX_COLUMN] == prompt_id, col].values[0]
                else:
                    logging.warning(f"Column '{col}' not found in the input DataFrame")
            results_list.append(result_dict)
            df_result = pd.DataFrame(results_list)
            try:
                df_result.to_pickle(os.path.join(DIR_RESULT, PKL_FILE_NAME), protocol=4)
                logging.info(f"Intermediate results saved for prompt {index}")
            except Exception as e:
                logging.error(f"Error saving intermediate results: {str(e)}")
                print(f"Error saving intermediate results: {str(e)}")
        
        # Rate limiting
        end = datetime.now()
        sleep_time = max(MAX_DELAY - (end - start).total_seconds(), 0)
        if sleep_time > 0:
            logging.info(f"Rate limiting: sleeping for {sleep_time:.1f} seconds")
            time.sleep(sleep_time)

    logging.info(f"Processing complete. {len(results_list)} out of {total_prompts_count} prompts processed successfully.")
    logging.info(f"Results saved to {DIR_RESULT}")
    print(f"Processing complete. {len(results_list)} out of {total_prompts_count} prompts processed successfully.")
    print(f"Results saved to {DIR_RESULT}")

def main():
    try:
        path_to_prompts = input("Enter the path to the PKL file containing prompts: ").strip()
        client = setup_anthropic_api()
        process_prompts(client, path_to_prompts)
    except Exception as e:
        logging.exception(f"An unexpected error occurred: {str(e)}")
        print(f"An unexpected error occurred. Please check the log file in {DIR_RESULT} for details.")
        sys.exit(1)

if __name__ == '__main__':
    main()
