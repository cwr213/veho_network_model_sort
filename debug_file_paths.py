# debug_file_paths.py - Quick script to check file locations and fix path issues
import os
from pathlib import Path
import pandas as pd


def check_file_locations():
    """Check current directory and find input files."""

    current_dir = Path.cwd()
    print(f"Current working directory: {current_dir}")
    print("\nFiles in current directory:")

    # List all Excel files
    excel_files = list(current_dir.glob("*.xlsx"))
    if excel_files:
        print("Excel files found:")
        for file in excel_files:
            print(f"  - {file.name}")
    else:
        print("No Excel files found in current directory")

        # Check common subdirectories
        common_dirs = ['data', 'inputs', 'input', '.']
        for dir_name in common_dirs:
            dir_path = current_dir / dir_name
            if dir_path.exists():
                excel_in_dir = list(dir_path.glob("*.xlsx"))
                if excel_in_dir:
                    print(f"\nExcel files found in {dir_name}/ directory:")
                    for file in excel_in_dir:
                        print(f"  - {file.name}")

    # Check if the specific input file exists anywhere nearby
    target_file = "veho_model_input_v4.xlsx"
    print(f"\nSearching for {target_file}...")

    # Search in current directory and subdirectories (max 2 levels deep)
    found_files = []
    for file_path in current_dir.rglob(target_file):
        if len(file_path.parts) - len(current_dir.parts) <= 2:  # Max 2 levels deep
            found_files.append(file_path)

    if found_files:
        print("Target file found at:")
        for file_path in found_files:
            print(f"  - {file_path}")
            # Check if it's a valid Excel file
            try:
                xl = pd.ExcelFile(file_path)
                sheets = xl.sheet_names
                print(f"    Valid Excel file with {len(sheets)} sheets")
                if len(sheets) <= 5:
                    print(f"    Sheets: {sheets}")
                else:
                    print(f"    First 5 sheets: {sheets[:5]}...")
            except Exception as e:
                print(f"    Error reading file: {e}")
    else:
        print(f"Target file '{target_file}' not found")

    return found_files


def suggest_command_fix(found_files):
    """Suggest the correct command to run."""
    if found_files:
        # Use the first found file
        input_file = found_files[0]

        print(f"\n{'=' * 60}")
        print("SUGGESTED COMMAND TO RUN:")
        print(f"{'=' * 60}")

        # Make path relative to current directory if possible
        try:
            relative_path = input_file.relative_to(Path.cwd())
            print(f'python run_v1.py --input "{relative_path}" --output_dir outputs')
        except ValueError:
            # Can't make relative, use absolute
            print(f'python run_v1.py --input "{input_file}" --output_dir outputs')

        print(f"{'=' * 60}")
    else:
        print("\n" + "=" * 60)
        print("INPUT FILE NOT FOUND - PLEASE:")
        print("=" * 60)
        print("1. Make sure veho_model_input_v4.xlsx is in the current directory, OR")
        print("2. Provide the full path to the file, OR")
        print("3. Copy the file to the current directory")
        print("=" * 60)


if __name__ == "__main__":
    print("🔍 Debugging file path issues...")
    found_files = check_file_locations()
    suggest_command_fix(found_files)