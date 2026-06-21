"""
CLAUDE BATCH PROCESSING SCRIPT - PRODUCTION READY
==================================================

CRITICAL FEATURES (Must work perfectly):
- Read pickle DataFrame (PROMPT_ID + PROMPT columns required)
- Submit batch to Claude API with proper request format
- Log batch ID immediately for recovery
- Poll until completion (auto-recovers from crashes)
- Download results as JSONL
- Merge results back into original DataFrame with ALL columns preserved
- Save final pickle with RESULT column added

STANDARD FEATURES (Important but not blocking):
- Logging to file and console
- Individual result .txt files
- Status tracking (succeeded/errored/expired/canceled)
- Error column for debugging

NICE-TO-HAVE FEATURES (Removed for simplicity):
- Multiple polling intervals
- Interactive resume menus
- Detailed progress bars
- Size validation warnings
"""

import os
import sys
import json
import time
from datetime import datetime
from anthropic import Anthropic
import pandas as pd

# ============================================================================
# CONFIGURATION - USER EDITABLE
# ============================================================================

SYSTEM_PROMPT_FILE = "SYSTEM_PROMPT.md"
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 32_000
PROMPT_INDEX_COLUMN = "PROMPT_ID"
PROMPT_COLUMN = "PROMPT"
POLL_INTERVAL_SECONDS = 60
TEMPERATURE = 1  # <---- CHANGE FOR MORE DETERMINISTIC RESPONSE FROM LLM.


# Time-stamped output directory
TIME_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
DIR_RESULT = f"results_{TIME_STAMP}"

# ============================================================================
# CRITICAL: Setup and logging
# ============================================================================

os.makedirs(DIR_RESULT, exist_ok=True)

# Simple logging to file and console
log_file = open(os.path.join(DIR_RESULT, "LOG.txt"), "a", encoding="utf-8")


def log(message):
    """CRITICAL: Log to both file and console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    log_file.write(log_message + "\n")
    log_file.flush()


# ============================================================================
# CRITICAL: Pre-flight checks
# ============================================================================


def load_system_prompt():
    """CRITICAL: Load system prompt from file with fallback."""
    # Try default location first
    if os.path.exists(SYSTEM_PROMPT_FILE):
        try:
            with open(SYSTEM_PROMPT_FILE, encoding="utf-8") as f:
                system_prompt = f.read()
            log(f"Loaded system prompt from {SYSTEM_PROMPT_FILE}")
            return system_prompt
        except Exception as e:
            log(f"ERROR: Could not read {SYSTEM_PROMPT_FILE} - {e}")
            sys.exit(1)

    # If not found, ask user for path
    log(f"System prompt file not found: {SYSTEM_PROMPT_FILE}")
    custom_path = input(f"Enter path to system prompt file: ").strip()

    try:
        with open(custom_path, encoding="utf-8") as f:
            system_prompt = f.read()
        log(f"Loaded system prompt from {custom_path}")
        return system_prompt
    except Exception as e:
        log(f"CRITICAL ERROR: Could not read system prompt file - {e}")
        sys.exit(1)


# ============================================================================
# CRITICAL: API Setup
# ============================================================================


def setup_api():
    """CRITICAL: Setup Anthropic API with key from env or user input."""
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        log("API key not found in environment variables")
        api_key = input("Paste your Anthropic API key: ").strip()

    try:
        client = Anthropic(api_key=api_key)
        # Test authentication
        client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": "test"}],
        )
        log("API authentication successful")
        return client
    except Exception as e:
        log(f"CRITICAL ERROR: API setup failed - {e}")
        sys.exit(1)


# ============================================================================
# CRITICAL: Batch submission
# ============================================================================


def submit_batch(client, df, system_prompt):
    """CRITICAL: Create and submit batch from DataFrame."""
    log(f"Creating batch from {len(df)} prompts")

    # Validate required columns
    if PROMPT_COLUMN not in df.columns or PROMPT_INDEX_COLUMN not in df.columns:
        log(f"CRITICAL ERROR: Required columns missing")
        log(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # Build batch requests
    requests = []
    for _, row in df.iterrows():
        requests.append(
            {
                "custom_id": str(row[PROMPT_INDEX_COLUMN]),
                "params": {
                    "model": CLAUDE_MODEL,
                    "max_tokens": MAX_TOKENS,
                    "system": system_prompt,
                    "temperature": TEMPERATURE,
                    "messages": [{"role": "user", "content": row[PROMPT_COLUMN]}],
                },
            }
        )

    # Submit batch
    try:
        batch = client.messages.batches.create(requests=requests)
        batch_id = batch.id

        # CRITICAL: Save batch ID immediately
        batch_id_file = os.path.join(DIR_RESULT, "BATCH_ID.txt")
        with open(batch_id_file, "w") as f:
            f.write(batch_id)

        log(f"BATCH SUBMITTED: {batch_id}")
        log(f"Batch ID saved to: {batch_id_file}")
        log(f"Status: {batch.processing_status}")
        log(f"Created: {batch.created_at}")
        log(f"Expires: {batch.expires_at}")

        return batch_id

    except Exception as e:
        log(f"CRITICAL ERROR: Batch submission failed - {e}")
        sys.exit(1)


# ============================================================================
# CRITICAL: Polling and completion
# ============================================================================


def poll_until_complete(client, batch_id):
    """CRITICAL: Poll batch status until complete. Auto-recovers from crashes."""
    log(f"Polling batch {batch_id} every {POLL_INTERVAL_SECONDS}s")

    while True:
        try:
            batch = client.messages.batches.retrieve(batch_id)
            status = batch.processing_status
            counts = batch.request_counts

            log(
                f"Status: {status} | "
                f"Succeeded: {counts.succeeded} | "
                f"Processing: {counts.processing} | "
                f"Errored: {counts.errored}"
            )

            if status == "ended":
                log("BATCH COMPLETE")
                return batch

            if status in ["canceling", "canceled"]:
                log("WARNING: Batch was canceled")
                return batch

            time.sleep(POLL_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            log("Polling interrupted by user. Batch continues on server.")
            log(f"Resume by running: poll_existing_batch('{batch_id}')")
            raise
        except Exception as e:
            log(f"Polling error (will retry): {e}")
            time.sleep(POLL_INTERVAL_SECONDS)


# ============================================================================
# CRITICAL: Results download and parsing
# ============================================================================


def download_results(client, batch_id):
    """CRITICAL: Download batch results as JSONL."""
    log(f"Downloading results for batch {batch_id}")

    try:
        results = []
        for result in client.messages.batches.results(batch_id):
            results.append(result)

        # Save raw JSONL
        jsonl_path = os.path.join(DIR_RESULT, "results.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result.model_dump()) + "\n")

        log(f"Downloaded {len(results)} results to {jsonl_path}")
        return results

    except Exception as e:
        log(f"CRITICAL ERROR: Results download failed - {e}")
        sys.exit(1)


def parse_results(results):
    """CRITICAL: Parse results into dict keyed by PROMPT_ID."""
    log("Parsing results")
    results_dict = {}

    for result in results:
        custom_id = result.custom_id
        result_type = result.result.type

        if result_type == "succeeded":
            text = result.result.message.content[0].text
            results_dict[custom_id] = {
                "RESULT": text,
                "STATUS": "succeeded",
                "ERROR": None,
            }

            # STANDARD: Save individual result file
            prefix = str(custom_id).zfill(4)
            with open(
                os.path.join(DIR_RESULT, f"{prefix}_result.txt"), "w", encoding="utf-8"
            ) as f:
                f.write(text)

        else:
            # errored, expired, or canceled
            error_msg = None
            if result_type == "errored":
                error = result.result.error
                error_msg = f"{error.type}: {error.message}"

            results_dict[custom_id] = {
                "RESULT": None,
                "STATUS": result_type,
                "ERROR": error_msg,
            }
            log(f"WARNING: Request {custom_id} status={result_type}")

    log(f"Parsed {len(results_dict)} results")
    return results_dict


# ============================================================================
# CRITICAL: Merge results with original DataFrame
# ============================================================================


def merge_results(df, results_dict):
    """CRITICAL: Merge results back into DataFrame preserving ALL columns."""
    log("Merging results with original DataFrame")

    # Create new columns
    result_col = []
    status_col = []
    error_col = []

    for _, row in df.iterrows():
        prompt_id = str(row[PROMPT_INDEX_COLUMN])

        if prompt_id in results_dict:
            result_col.append(results_dict[prompt_id]["RESULT"])
            status_col.append(results_dict[prompt_id]["STATUS"])
            error_col.append(results_dict[prompt_id]["ERROR"])
        else:
            # Should never happen but handle gracefully
            result_col.append(None)
            status_col.append("missing")
            error_col.append("No result returned from API")
            log(f"WARNING: No result for PROMPT_ID {prompt_id}")

    # Add columns to DataFrame
    df_final = df.copy()
    df_final["RESULT"] = result_col
    df_final["STATUS"] = status_col
    df_final["ERROR"] = error_col

    log(f"Merged results. Final shape: {df_final.shape}")
    log(f"Final columns: {list(df_final.columns)}")

    return df_final


# ============================================================================
# CRITICAL: Save final results
# ============================================================================


def save_final_results(df_final):
    """CRITICAL: Save final DataFrame as pickle."""
    output_path = os.path.join(DIR_RESULT, "RESULTS.pkl")

    try:
        df_final.to_pickle(output_path, protocol=4)
        log(f"FINAL RESULTS SAVED: {output_path}")

        # Print summary
        succeeded = (df_final["STATUS"] == "succeeded").sum()
        errored = (df_final["STATUS"] == "errored").sum()
        expired = (df_final["STATUS"] == "expired").sum()
        canceled = (df_final["STATUS"] == "canceled").sum()

        log("=" * 60)
        log("SUMMARY")
        log("=" * 60)
        log(f"Total prompts: {len(df_final)}")
        log(f"Succeeded: {succeeded}")
        log(f"Errored: {errored}")
        log(f"Expired: {expired}")
        log(f"Canceled: {canceled}")
        log(f"Results directory: {DIR_RESULT}")
        log("=" * 60)

    except Exception as e:
        log(f"CRITICAL ERROR: Failed to save results - {e}")
        sys.exit(1)


# ============================================================================
# MAIN WORKFLOW
# ============================================================================


def process_new_batch(pkl_path):
    """CRITICAL: Main workflow for new batch."""
    log("=" * 60)
    log("STARTING NEW BATCH")
    log("=" * 60)

    # Pre-flight check: Load system prompt
    system_prompt = load_system_prompt()

    # Load DataFrame
    try:
        df = pd.read_pickle(pkl_path)
        log(f"Loaded DataFrame: {len(df)} rows")
        log(f"Columns: {list(df.columns)}")
    except Exception as e:
        log(f"CRITICAL ERROR: Failed to load pickle - {e}")
        sys.exit(1)

    # Setup API
    client = setup_api()

    # Submit batch
    batch_id = submit_batch(client, df, system_prompt)

    # Poll until complete
    completed_batch = poll_until_complete(client, batch_id)

    # Download results
    results = download_results(client, batch_id)

    # Parse results
    results_dict = parse_results(results)

    # Merge with DataFrame
    df_final = merge_results(df, results_dict)

    # Save final results
    save_final_results(df_final)

    log("BATCH PROCESSING COMPLETE")
    return df_final


def resume_existing_batch(batch_id=None):
    """Resume polling an existing batch (for crash recovery)."""
    log("=" * 60)
    log("RESUMING EXISTING BATCH")
    log("=" * 60)

    # Get batch ID
    if not batch_id:
        batch_id = input("Enter Batch ID: ").strip()

    log(f"Resuming batch: {batch_id}")

    # Setup API
    client = setup_api()

    # Check batch status
    try:
        batch = client.messages.batches.retrieve(batch_id)
        log(f"Current status: {batch.processing_status}")
    except Exception as e:
        log(f"CRITICAL ERROR: Could not retrieve batch - {e}")
        sys.exit(1)

    # If already complete, just download results
    if batch.processing_status == "ended":
        log("Batch already complete, downloading results")
    else:
        # Continue polling
        batch = poll_until_complete(client, batch_id)

    # Download results
    results = download_results(client, batch_id)

    # Parse results
    results_dict = parse_results(results)

    log(f"Downloaded {len(results_dict)} results")
    log(
        "To merge with original DataFrame, load your pickle and call merge_results(df, results_dict)"
    )

    return results_dict


# ============================================================================
# ENTRY POINT
# ============================================================================


def main():
    print("=" * 60)
    print("CLAUDE BATCH PROCESSING")
    print("=" * 60)
    print("1. Process new batch")
    print("2. Resume existing batch")
    print("=" * 60)

    choice = input("Choose option (1 or 2): ").strip()

    try:
        if choice == "1":
            pkl_path = input("Enter path to pickle file: ").strip()
            process_new_batch(pkl_path)

        elif choice == "2":
            batch_id_input = input(
                "Enter Batch ID (or press Enter to load from BATCH_ID.txt): "
            ).strip()

            if not batch_id_input:
                # Try to load from most recent results directory
                try:
                    batch_id_path = os.path.join(DIR_RESULT, "BATCH_ID.txt")
                    with open(batch_id_path) as f:
                        batch_id_input = f.read().strip()
                    log(f"Loaded batch ID from {batch_id_path}")
                except:
                    log("ERROR: Could not find BATCH_ID.txt")
                    sys.exit(1)

            resume_existing_batch(batch_id_input)

        else:
            print("Invalid choice")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nInterrupted. Your batch continues processing on Claude's servers.")
        print(f"Check BATCH_ID.txt in {DIR_RESULT} to resume later.")
        sys.exit(0)
    except Exception as e:
        log(f"UNEXPECTED ERROR: {e}")
        import traceback

        log(traceback.format_exc())
        sys.exit(1)
    finally:
        log_file.close()


if __name__ == "__main__":
    main()
