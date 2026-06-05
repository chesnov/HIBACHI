#!/bin/bash
# Ensure we are running in the HIBACHI folder
cd "$(dirname "$0")"

echo "Checking for updates..."
git fetch

# Check how many commits behind the remote branch we are
BEHIND=$(git rev-list HEAD..@{u} --count 2>/dev/null || echo 0)

if [ "$BEHIND" -gt 0 ]; then
    echo "========================================================"
    echo " [NOTICE] HIBACHI is behind by $BEHIND updates!"
    echo "========================================================"
    read -p "Would you like to download and install the update now? (Y/n): " -n 1 -r
    echo ""
    # Defaults to Yes if they just hit Enter
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        echo ""
        echo "Pulling latest code..."
        git pull
        
        echo ""
        echo "Updating conda environment (this may take a minute)..."
        # Hook conda into the script so it can run env update
        eval "$(conda shell.bash hook)"
        conda env update -f environment.yaml --prune
    fi
else
    echo "HIBACHI is up to date!"
fi

echo ""
echo "Launching HIBACHI GUI..."
# Hook conda into the script to allow 'conda activate'
eval "$(conda shell.bash hook)"
conda activate hibachi
python segment.py