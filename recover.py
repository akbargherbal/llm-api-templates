# recover.py
import os
import sys
import argparse
import pandas as pd
import subprocess
import re

# --- CONFIGURATION ---
MAIN_SCRIPT_NAME = "GAMMA_gemini-api-template.py"
RETRY_JOB_FILE = "RETRY_INPUT.pkl"
PROMPT_ID_COLUMN = "PROMPT_ID"


def main():
    """
    Main function to orchestrate the recovery process.
    """
    parser = argparse.ArgumentParser(
        description="A Post-Mortem Recovery Agent for GAMMA_gemini-api-template.py. "
        "This script identifies missing prompts from an incomplete run, "
        "re-runs them, and merges the results.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the original PKL file containing all prompts (e.g., INPUT_PROMPTS.pkl).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the incomplete RESULTS.pkl file from the failed run (e.g., results_20250722_095141/RESULTS.pkl).",
    )
    args = parser.parse_args()

    # --- Step 1: Analyze Discrepancy ---
    print(f"[*] Analyzing failed run...")
    print(f"    - Original Input: {args.input}")
    print(f"    - Failed Output: {args.output}")

    try:
        df_original_prompts = pd.read_pickle(args.input)
        df_partial_results = pd.read_pickle(args.output)
    except FileNotFoundError as e:
        print(f"[!] Error: Could not open a file. {e}", file=sys.stderr)
        sys.exit(1)

    original_ids = set(df_original_prompts[PROMPT_ID_COLUMN])
    completed_ids = set(df_partial_results[PROMPT_ID_COLUMN])
    missing_ids = list(original_ids - completed_ids)

    if not missing_ids:
        print(
            "\n[+] Success: No missing prompts found. The output file is already complete."
        )
        sys.exit(0)

    print(f"\n[*] Found {len(missing_ids)} missing prompt(s): {sorted(missing_ids)}")

    # --- Step 2: Create Retry Job ---
    print(f"[*] Creating a new retry job...")
    df_retry = df_original_prompts[
        df_original_prompts[PROMPT_ID_COLUMN].isin(missing_ids)
    ]
    df_retry.to_pickle(RETRY_JOB_FILE)
    print(f"    - Temporary retry file created: {RETRY_JOB_FILE}")

    # --- Step 3: Execute Recovery Run ---
    print("\n[*] Invoking the main script to process the retry job...")
    try:
        # We need to provide the input file name via stdin, as the script expects it.
        # We capture stdout to find the name of the new results directory.
        process = subprocess.run(
            [sys.executable, MAIN_SCRIPT_NAME],
            input=f"{RETRY_JOB_FILE}\n",
            text=True,
            capture_output=True,
            check=True,
        )

        # Find the new results directory from the script's output
        output_lines = process.stdout
        match = re.search(r"Results saved to (results_\d{8}_\d{6})", output_lines)
        if not match:
            print(
                "[!] Critical Error: Could not determine the results directory from the script output.",
                file=sys.stderr,
            )
            print(
                "--- SCRIPT STDOUT ---",
                process.stdout,
                "--- SCRIPT STDERR ---",
                process.stderr,
                sep="\n",
            )
            sys.exit(1)

        new_results_dir = match.group(1)
        path_to_new_results = os.path.join(new_results_dir, "RESULTS.pkl")
        print(f"[*] Recovery run complete. New results are in: {path_to_new_results}")

    except subprocess.CalledProcessError as e:
        print(f"[!] Error: The recovery run failed.", file=sys.stderr)
        print(f"    - Return Code: {e.returncode}")
        print(f"    - STDOUT: {e.stdout}")
        print(f"    - STDERR: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print(
            f"[!] Error: Main script '{MAIN_SCRIPT_NAME}' not found in the current directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Step 4: Merge and Finalize ---
    print("\n[*] Merging results...")
    try:
        df_new_results = pd.read_pickle(path_to_new_results)
        df_complete = pd.concat([df_partial_results, df_new_results], ignore_index=True)
        df_complete = df_complete.sort_values(by=PROMPT_ID_COLUMN).reset_index(
            drop=True
        )

        # Overwrite the original partial results file with the complete one
        df_complete.to_pickle(args.output)

        print(
            f"[*] Merge successful. The complete dataset now contains {len(df_complete)} records."
        )
        print(f"    - Updated results file: {args.output}")

    except Exception as e:
        print(f"[!] Error during the final merge step: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Step 5: Cleanup ---
    finally:
        if os.path.exists(RETRY_JOB_FILE):
            os.remove(RETRY_JOB_FILE)
            print(f"[*] Cleanup complete. Removed temporary file: {RETRY_JOB_FILE}")


if __name__ == "__main__":
    main()
