# LLM API Templates

This repository contains Python script templates for interacting with various Large Language Model (LLM) APIs. These templates are designed to provide a consistent structure for processing multiple prompts and handling responses across different LLM providers.

## Available Templates

1. `BETA_gpt4-api-template.py`: Template for OpenAI's GPT-4 API
2. `BETA_claude-api-template.py`: Template for Anthropic's Claude API
3. `BETA_gemini-api-template.py`: Template for Google's Gemini Pro API

## Features

- Consistent input format using pandas DataFrame (pickle file)
- Robust error handling and logging
- Rate limiting to respect API constraints
- Intermediate saving of results to prevent data loss
- Flexible handling of additional data columns

## Usage

1. Prepare your input data:
   - Create a pandas DataFrame with at least two columns: 'PROMPT_ID' and 'PROMPT'
   - Save the DataFrame as a pickle file

2. Install required dependencies:
   ```
   pip install pandas openai anthropic google-generativeai
   ```

3. Run the desired script:
   ```
   python BETA_<api>-api-template.py
   ```

4. When prompted, enter:
   - The path to your input pickle file
   - Your API key for the respective service

5. The script will process your prompts and save the results in a time-stamped directory.

## Configuration

Each script has several configurable parameters at the top:

- `MAX_TOKENS`: Maximum number of tokens for the API response
- `MAX_DELAY`: Maximum delay between API calls (for rate limiting)
- `ADDITIONAL_COLUMNS`: List of additional columns from the input DataFrame to include in the results

Adjust these as needed for your specific use case.

## Output

The scripts generate three types of output:

1. Individual text files for each prompt response
2. A pickle file containing all results as a DataFrame
3. A log file with processing details and any errors

All outputs are saved in a time-stamped directory for easy tracking.

## Notes

- These templates are designed for personal use and may need additional security measures for production environments.
- Always keep your API keys secure and never commit them to version control.
- Regularly check the API documentation for any changes in endpoints or parameters.

## Future Improvements

- Add support for more LLM APIs
- Implement async processing for improved performance
- Add command-line arguments for easier configuration

Remember to update this README as you make changes to the templates or add new features!
