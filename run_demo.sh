#!/bin/bash

echo "========================================"
echo "Ethiopian E-commerce NER System Demo"
echo "========================================"
echo

echo "Setting up environment..."
export PYTHONPATH="$(pwd)/src"

echo
echo "Running comprehensive system demonstration..."
echo "This will showcase all 6 tasks with sample data."
echo

python scripts/demo.py

echo
echo "Demo completed! Check the reports/ directory for detailed results."
echo
echo "To run individual components:"
echo "  - Streamlit Dashboard: streamlit run src/dashboard/streamlit_app.py"
echo "  - Unit Tests: python -m pytest tests/unit/ -v"
echo "  - Integration Tests: python -m pytest tests/integration/ -v"
echo "  - Docker: docker-compose up"
echo