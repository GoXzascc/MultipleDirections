#!/usr/bin/env bash
# Run all analysis scripts in sequence
# Exit on error, undefined variables, and pipe failures
set -euo pipefail

# Clear any conflicting VIRTUAL_ENV variable to avoid uv warnings
unset VIRTUAL_ENV

# Color output for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored messages
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get script start time
SCRIPT_START=$(date +%s)

log_info "Starting analysis pipeline at $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Array of scripts to run
SCRIPTS=(
    "src/extract_concepts.py"
    # "src/curvature.py"
    # "src/direction_alignment.py"
    # "src/trajectory_smoothness.py"
    # "src/norm_decomposition.py"
)

# Run each script
for script in "${SCRIPTS[@]}"; do
    log_info "Running: $script"
    START=$(date +%s)
    
    if uv run "$script"; then
        END=$(date +%s)
        DURATION=$((END - START))
        log_success "Completed: $script (${DURATION}s)"
    else
        log_error "Failed: $script"
        exit 1
    fi
    echo ""
done

# Calculate total time
SCRIPT_END=$(date +%s)
TOTAL_DURATION=$((SCRIPT_END - SCRIPT_START))
MINUTES=$((TOTAL_DURATION / 60))
SECONDS=$((TOTAL_DURATION % 60))

echo ""
log_success "All scripts completed successfully!"
log_info "Total time: ${MINUTES}m ${SECONDS}s"
log_info "Finished at $(date '+%Y-%m-%d %H:%M:%S')"