# recover.py - Improved Recovery Agent
import os
import sys
import argparse
import pandas as pd
import subprocess
import re
from pathlib import Path

# --- CONFIGURATION ---
MAIN_SCRIPT_NAME = "GAMMA_gemini-api-template.py"
RETRY_JOB_FILE = Path("RETRY_INPUT.pkl")
PROMPT_ID_COLUMN = "PROMPT_ID"
MERGED_FILENAME_SUFFIX = ".merged.pkl"


def find_missing_ids(original_prompts_path: Path, partial_results_path: Path) -> tuple:
    """Compares original prompts and partial results to find missing prompt IDs."""
    print("[*] Analyzing failed run...")
    print(f"    - Original Input: {original_prompts_path}")
    print(f"    - Incomplete Output: {partial_results_path}")

    df_original = pd.read_pickle(original_prompts_path)
    df_partial = pd.read_pickle(partial_results_path)

    original_ids = set(df_original[PROMPT_ID_COLUMN])
    completed_ids = set(df_partial[PROMPT_ID_COLUMN])
    missing_ids = sorted(list(original_ids - completed_ids))

    if not missing_ids:
        print(
            "\n[+] Success: No missing prompts found. The output file is already complete."
        )
        sys.exit(0)

    print(f"\n[*] Found {len(missing_ids)} missing prompt(s): {missing_ids}")
    return missing_ids, df_original, df_partial


def create_retry_job(df_original: pd.DataFrame, missing_ids: list):
    """Creates a new PKL file containing only the prompts that need to be retried."""
    print(f"[*] Creating a new retry job...")
    df_retry = df_original[df_original[PROMPT_ID_COLUMN].isin(missing_ids)]
    df_retry.to_pickle(RETRY_JOB_FILE)
    print(f"    - Temporary retry file created: {RETRY_JOB_FILE}")


def execute_recovery_run() -> Path:
    """
    Invokes the main script to process the retry job file.
    This function is coupled to the main script's specific I/O behavior:
    1. It expects an interactive `input()` prompt for the filename.
    2. It parses a human-readable log message to find the results directory.
    """
    print("\n[*] Invoking the main script to process the retry job...")
    try:
        # Capture stdout and stderr separately for better error diagnostics
        process = subprocess.run(
            [sys.executable, MAIN_SCRIPT_NAME],
            input=f"{RETRY_JOB_FILE}\n",
            text=True,
            capture_output=True,
            check=True,
        )

        # Find the new results directory by parsing the script's output
        output_text = process.stdout
        match = re.search(r"Results saved to (results_\d{8}_\d{6})", output_text)
        if not match:
            print(
                "[!] Critical Error: Could not determine the results directory from the script output.",
                file=sys.stderr,
            )
            print("--- SCRIPT STDOUT ---")
            print(process.stdout)
            print("--- SCRIPT STDERR ---")
            print(process.stderr)
            sys.exit(1)

        new_results_dir = Path(match.group(1))
        path_to_new_results = new_results_dir / "RESULTS.pkl"
        print(f"[*] Recovery run complete. New results are in: {path_to_new_results}")
        return path_to_new_results

    except subprocess.CalledProcessError as e:
        print("[!] Error: The recovery run failed.", file=sys.stderr)
        print(f"    - Return Code: {e.returncode}")
        print(f"--- STDOUT ---\n{e.stdout}")
        print(f"--- STDERR ---\n{e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print(
            f"[!] Error: Main script '{MAIN_SCRIPT_NAME}' not found in the current directory.",
            file=sys.stderr,
        )
        sys.exit(1)


def merge_and_finalize(
    df_partial: pd.DataFrame, new_results_path: Path, original_output_path: Path
):
    """Merges the partial and new results into a final, complete file."""
    print("\n[*] Merging results...")
    df_new_results = pd.read_pickle(new_results_path)
    df_complete = pd.concat([df_partial, df_new_results], ignore_index=True)
    df_complete = df_complete.sort_values(by=PROMPT_ID_COLUMN).reset_index(drop=True)

    # Safer: Write to a new file instead of overwriting the original partial results
    merged_output_path = original_output_path.with_name(
        original_output_path.stem + MERGED_FILENAME_SUFFIX
    )
    df_complete.to_pickle(merged_output_path)

    print(
        f"[*] Merge successful. The complete dataset now contains {len(df_complete)} records."
    )
    print(f"    - Final merged results saved to: {merged_output_path}")
    print(f"    - Original partial results preserved at: {original_output_path}")


def main():
    """Main function to orchestrate the recovery process."""
    parser = argparse.ArgumentParser(
        description="Recovers from a failed or incomplete run of GAMMA_gemini-api-template.py.\n"
        "It finds which prompts were not processed, runs them in a new job, and\n"
        "merges the new results with the partial results from the failed run.",
        epilog="--- Example Usage ---\n"
        "Imagine your main script failed halfway through processing your prompts.\n\n"
        "  1. Your original, complete input file was: `ALL_PROMPTS.pkl`\n"
        "  2. The failed script created a directory: `results_20231027_123456/`\n"
        "  3. Inside that directory is an incomplete results file: `RESULTS.pkl`\n\n"
        "To recover, you would run this command:\n"
        "  python recover.py --input ALL_PROMPTS.pkl --output results_20231027_123456/RESULTS.pkl\n\n"
        "The script will then create a final, merged file named `results_20231027_123456/RESULTS.merged.pkl`\n"
        "while preserving your original incomplete results as a backup.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the ORIGINAL, complete PKL file that contains ALL prompts.\n"
        "Example: `ALL_PROMPTS.pkl`",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the INCOMPLETE `RESULTS.pkl` file from the FAILED run.\n"
        "This file is located inside a `results_...` directory.\n"
        "Example: `results_20231027_123456/RESULTS.pkl`",
    )
    args = parser.parse_args()

    original_prompts_path = Path(args.input)
    partial_results_path = Path(args.output)

    try:
        # Step 1: Analyze discrepancy
        missing_ids, df_original, df_partial = find_missing_ids(
            original_prompts_path, partial_results_path
        )

        # Step 2: Create a temporary job file with only the missing prompts
        create_retry_job(df_original, missing_ids)

        # Step 3: Execute the main script on the temporary job file
        new_results_path = execute_recovery_run()

        # Step 4: Merge the original partial results with the new results
        merge_and_finalize(df_partial, new_results_path, partial_results_path)

    except FileNotFoundError as e:
        print(f"[!] Error: Could not open a file. {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[!] An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Step 5: Cleanup the temporary file
        if RETRY_JOB_FILE.exists():
            RETRY_JOB_FILE.unlink()
            print(f"[*] Cleanup complete. Removed temporary file: {RETRY_JOB_FILE}")


if __name__ == "__main__":
    main()
