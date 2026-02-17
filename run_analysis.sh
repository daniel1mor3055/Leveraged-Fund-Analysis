#!/bin/bash

# Leveraged Fund Analysis - Quick Run Script
# Activates venv and runs the complete analysis pipeline

set -e

echo "=================================="
echo "Leveraged Fund Analysis"
echo "=================================="
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Check if dependencies are installed
if [ ! -f "venv/.deps_installed" ]; then
    echo "Installing dependencies..."
    venv/bin/pip3 install -q -r requirements.txt
    touch venv/.deps_installed
fi

# Run analysis
echo "Running analysis pipeline..."
echo ""
venv/bin/python3 src/main.py

echo ""
echo "=================================="
echo "Analysis complete!"
echo "=================================="
echo ""
echo "View results:"
echo "  - Summary: output/results/summary_report.txt"
echo "  - Charts: output/figures/"
echo "  - Data: data/processed/"
echo ""
