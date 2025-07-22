#!/bin/bash

PYTHON_VERSIONS=("3.10" "3.11" "3.12" "3.13")
PYTHON_SCRIPT="reproducibility.py"
PACKAGE_TO_INSTALL="baybe==0.13.2"


# --- Script Logic ---

# First, check if 'uv' is installed and available in the system's PATH.
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' is not installed or not in your PATH."
    echo "Please install it to continue. See: https://github.com/astral-sh/uv"
    exit 1
fi

# Check if the target Python script exists in the current directory.
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT' not found."
    echo "Please make sure it is in the same directory as this bash script."
    exit 1
fi

# Loop over each specified Python version.
for VERSION in "${PYTHON_VERSIONS[@]}"; do
    echo "Python $VERSION"
    for i in {1..3}; do
        # The '--quiet' flag suppresses uv's installation and resolution messages.
        # The command to be run in the environment follows the '--'.
        EXEC_RESULT=$(uv run --quiet --frozen --python $VERSION -- python $PYTHON_SCRIPT)

        # Check if the command executed successfully and produced output.
        if [ -n "$EXEC_RESULT" ]; then
            echo "      Run $i:"
            echo "      $EXEC_RESULT"
        else
            echo "      Run $i: Failed to execute script with Python $VERSION."
            # If one run fails, we break the inner loop as subsequent runs
            # for this Python version will likely also fail.
            break
        fi
    done

    echo "" # Add a newline for better separation in the output.
done