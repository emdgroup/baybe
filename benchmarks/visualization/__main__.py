"""Transfer Learning convergence visualization module for benchmarks.

Usage:
  python -m benchmarks.visualization --type per_source --file-dir results_folder
  python -m benchmarks.visualization --type per_model --file-names file1.json file2.json
"""

import argparse
import sys
from pathlib import Path

# Add the parent directory to the path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    """Visualize Transfer Learning convergence benchmarks from JSON files."""
    parser = argparse.ArgumentParser(
        description="Visualize Transfer Learning convergence benchmark results from JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python -m benchmarks.visualization --type per_source file.json

  # Multiple specific files
  python -m benchmarks.visualization --type per_model --file-names file1.json file2.json

  # All files matching pattern
  python -m benchmarks.visualization --type per_source --file-names *_result.json

  # All JSON files in directory and subdirectories (recommended)
  python -m benchmarks.visualization --type per_source --file-dir results_folder
  python -m benchmarks.visualization --type per_model --file-dir 2025_12_23_benchmark_pos_index_kernel_github_action
        """,
    )

    parser.add_argument(
        "--type",
        choices=["per_source", "per_model"],
        required=True,
        help="Type of TL convergence visualization: 'per_source' (columns=source%, legend=suffixes) or 'per_model' (columns=suffixes, legend=source%)",
    )

    # Support both single file (positional), multiple files (--file-names), and directory (--file-dir)
    parser.add_argument("file", nargs="?", help="Single JSON result file to visualize")
    parser.add_argument(
        "--file-names", nargs="+", help="List of JSON result files to visualize"
    )
    parser.add_argument(
        "--file-dir", help="Directory to recursively search for JSON result files"
    )

    args = parser.parse_args()

    # Determine which files to process
    options_count = sum([args.file is not None, args.file_names is not None, args.file_dir is not None])
    if options_count > 1:
        print("Error: Cannot specify multiple input options (file, --file-names, --file-dir).")
        sys.exit(1)
    elif args.file:
        file_paths = [args.file]
    elif args.file_names:
        file_paths = args.file_names
    elif args.file_dir:
        # Recursively find all .json files in the directory
        dir_path = Path(args.file_dir)
        if not dir_path.exists():
            print(f"Error: Directory '{args.file_dir}' does not exist.")
            sys.exit(1)
        if not dir_path.is_dir():
            print(f"Error: '{args.file_dir}' is not a directory.")
            sys.exit(1)

        file_paths = [str(f) for f in dir_path.rglob("*.json")]
        if not file_paths:
            print(f"Error: No JSON files found in directory '{args.file_dir}'.")
            sys.exit(1)

        print(f"Found {len(file_paths)} JSON files in directory '{args.file_dir}'")
        # Sort for consistent processing order
        file_paths.sort()
    else:
        print("Error: Must specify either a file, use --file-names option, or use --file-dir option.")
        parser.print_help()
        sys.exit(1)

    # Import and call the appropriate visualization function
    successful_files = []
    failed_files = []

    # Import the appropriate visualization function
    if args.type == "per_source":
        from benchmarks.visualization.visualize_tl_convergence_benchmark_per_source import (
            visualize_tl_convergence_per_source,
        )
        visualize_func = visualize_tl_convergence_per_source
        viz_name = "TL convergence per source"
    elif args.type == "per_model":
        from benchmarks.visualization.visualize_tl_convergence_benchmark_per_model import (
            visualize_tl_convergence_per_model,
        )
        visualize_func = visualize_tl_convergence_per_model
        viz_name = "TL convergence per model"

    # Process all files
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
            output_path = visualize_func(file_path)
            print(f"Successfully created {viz_name} visualization for '{file_path}'!")
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
