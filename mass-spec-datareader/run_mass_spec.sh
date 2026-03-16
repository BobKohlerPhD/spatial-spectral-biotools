#!/bin/bash

# Navigate to the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "=== Mass Spec Data Analysis Pipeline ==="

# Check for .NET 8
if ! command -v dotnet &> /dev/null
then
    echo "Error: .NET 8 Runtime not found."
    echo "Please download it from: https://dotnet.microsoft.com/en-us/download/dotnet/8.0"
    read -p "Press enter to exit..."
    exit 1
fi

# Set up virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Setting up Python virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

# Run the processor
# You can drag and drop an input directory here or edit the script
if [ -z "$1" ]; then
    echo "Usage: ./run_mass_spec.sh [INPUT_DIRECTORY]"
    read -p "Enter input directory path: " INPUT_DIR
else
    INPUT_DIR="$1"
fi

python3 processor.py -i "$INPUT_DIR"

echo ""
echo "Analysis complete."
read -p "Press enter to exit..."
