#!/usr/bin/env bash

set -e

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    if command_exists python3; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
        PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
        if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]; }; then
            echo "❌ Python 3.10 or higher is required. Current version: $PYTHON_VERSION"
            exit 1
        else
            echo "✅ Detected Python version: $PYTHON_VERSION"
        fi
    else
        echo "❌ python3 is not installed. Please install Python 3.10 or higher."
        exit 1
    fi
}

echo "🔧 Setting up the development environment..."

# 1. Check Python version
check_python_version

# 2. Install uv if not present
if ! command_exists uv; then
    echo "📦 'uv' not found. Installing..."
    curl -Ls https://astral.sh/uv/install.sh | sh
else
    echo "✅ 'uv' is already installed."
fi

# 3. Install pre-commit if not present
if ! command_exists pre-commit; then
    echo "📦 'pre-commit' not found. Installing..."
    pip install pre-commit
else
    echo "✅ 'pre-commit' is already installed."
fi

# 4. Create virtual environment with uv
echo "🐍 Creating virtual environment with 'uv'..."
uv venv --python 3.11 --prompt "divsuite_env"

# 5. Synchronize dependencies using uv
echo "🔄 Synchronizing dependencies with 'uv'..."
uv sync --all-extras --dev

# 6. Install pre-commit hooks
echo "🔗 Installing pre-commit hooks..."
pre-commit install --install-hooks

echo ""
echo "✅ Setup complete!"
echo ""
echo "👉 To activate the virtual environment, run:"
echo "   source .venv/bin/activate"
echo ""
echo "Happy coding! 🚀"
