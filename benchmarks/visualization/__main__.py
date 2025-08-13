#!/usr/bin/env python3
"""Unified visualization module for benchmarks.

Usage:
  python -m benchmarks.visualize --type regression --file-names file1.json file2.json
  python -m benchmarks.visualize --type convergence --file-names file1.json file2.json
"""

import argparse
import sys
from pathlib import Path

# Add the parent directory to the path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    """Visualize convergence and regression benchmarks from JSON files."""
    parser = argparse.ArgumentParser(
        description="Visualize benchmark results from JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m benchmarks.visualize --type regression file.json
  python -m benchmarks.visualize --type regression --file-names file1.json file2.json
  python -m benchmarks.visualize --type convergence --file-names *.json
        """,
    )

    parser.add_argument(
        "--type",
        choices=["regression", "convergence"],
        required=True,
        help="Type of benchmark visualization to generate",
    )

    # Support both single file (positional) and multiple files (--file-names)
    parser.add_argument("file", nargs="?", help="Single JSON result file to visualize")
    parser.add_argument(
        "--file-names", nargs="+", help="List of JSON result files to visualize"
    )

    args = parser.parse_args()

    # Determine which files to process
    if args.file and args.file_names:
        print("Error: Cannot specify both positional file and --file-names option.")
        sys.exit(1)
    elif args.file:
        file_paths = [args.file]
    elif args.file_names:
        file_paths = args.file_names
    else:
        print("Error: Must specify either a file or use --file-names option.")
        parser.print_help()
        sys.exit(1)

    # Import and call the appropriate visualization function
    if args.type == "regression":
        from benchmarks.visualization.visualize_regression_benchmark import (
            visualize_regression_benchmark,
        )

        successful_files = []
        failed_files = []

        for file_path in file_paths:
            if not Path(file_path).exists():
                print(f"Error: File '{file_path}' does not exist.")
                failed_files.append(file_path)
                continue

            if not file_path.endswith(".json"):
                print(f"Error: File '{file_path}' is not a JSON file.")
                failed_files.append(file_path)
                continue

            try:
                visualize_regression_benchmark(file_path)
                print(
                    f"Successfully created regressionvisualization for '{file_path}'!"
                )
                successful_files.append(file_path)
            except Exception as e:
                print(f"Error creating visualization for '{file_path}': {e}")
                failed_files.append(file_path)

        # Summary
        print("\nSummary:")
        print(f"  Successfully processed: {len(successful_files)} files")
        print(f"  Failed: {len(failed_files)} files")

        if failed_files:
            print(f"  Failed files: {', '.join(failed_files)}")
            sys.exit(1)

    elif args.type == "convergence":
        from benchmarks.visualization.visualize_convergence_benchmark import (
            visualize_convergence_benchmark,
        )

        successful_files = []
        failed_files = []

        for file_path in file_paths:
            if not Path(file_path).exists():
                print(f"Error: File '{file_path}' does not exist.")
                failed_files.append(file_path)
                continue

            if not file_path.endswith(".json"):
                print(f"Error: File '{file_path}' is not a JSON file.")
                failed_files.append(file_path)
                continue

            try:
                output_path = visualize_convergence_benchmark(file_path)
                print(
                    f"Successfully created convergencevisualization for '{file_path}'!"
                )
                print(f"Output saved as: {output_path}")
                successful_files.append(file_path)
            except Exception as e:
                print(f"Error creating visualization for '{file_path}': {e}")
                failed_files.append(file_path)

        # Summary
        print("\nSummary:")
        print(f"  Successfully processed: {len(successful_files)} files")
        print(f"  Failed: {len(failed_files)} files")

        if failed_files:
            print(f"  Failed files: {', '.join(failed_files)}")
            sys.exit(1)


if __name__ == "__main__":
    main()
