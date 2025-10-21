#!/bin/bash
#
# Complete Analysis Workflow
# Runs all experiments needed for the report
#

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          HATE SPEECH DETECTION - COMPLETE ANALYSIS WORKFLOW              ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if fine-tuned model exists
if [ ! -d "models/fine_tuned_balanced" ] && [ ! -d "models/fine_tuned" ]; then
    echo -e "${YELLOW}⚠️  No fine-tuned model found!${NC}"
    echo -e "${YELLOW}   Run this first:${NC}"
    echo -e "${YELLOW}   python src/fine_tune_model.py --balance-classes --epochs 6 --compare${NC}"
    echo ""
    read -p "Do you want to run fine-tuning now? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}[0/4] Fine-tuning model...${NC}"
        python src/fine_tune_model.py --balance-classes --epochs 6 --compare --output models/fine_tuned_balanced
    else
        echo -e "${RED}Skipping fine-tuning. Using base model only.${NC}"
    fi
else
    echo -e "${GREEN}✓ Fine-tuned model found${NC}"
fi

# Determine which models to compare
MODELS="all-MiniLM-L6-v2"
if [ -d "models/fine_tuned_balanced" ]; then
    MODELS="$MODELS models/fine_tuned_balanced"
elif [ -d "models/fine_tuned" ]; then
    MODELS="$MODELS models/fine_tuned"
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}[1/4] Comparing Embedding Models${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "Models: $MODELS simcse-bert"
echo "Sample size: 3000"
echo "Clusters: 12"
echo ""

python src/embedding_comparison.py \
  --models $MODELS simcse-bert \
  --sample-size 3000 \
  --n-clusters 12

echo -e "${GREEN}✓ Model comparison complete${NC}"
echo -e "  Results saved to: outputs/comparison/"
echo ""

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}[2/4] Running Semi-Supervised Learning${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "Using config.yaml settings"
echo "Running self-training..."
echo ""

# Backup current config
cp config.yaml config.yaml.backup

# Update config to use fine-tuned model if available
if [ -d "models/fine_tuned_balanced" ]; then
    echo "Using fine-tuned model: models/fine_tuned_balanced"
    # Note: You may need to manually update config.yaml before running this
elif [ -d "models/fine_tuned" ]; then
    echo "Using fine-tuned model: models/fine_tuned"
fi

python src/main.py --config config.yaml || {
    echo -e "${YELLOW}⚠️  Main pipeline failed. Check errors above.${NC}"
    echo -e "${YELLOW}   Continuing with remaining steps...${NC}"
}

echo -e "${GREEN}✓ Semi-supervised learning complete${NC}"
echo -e "  Results saved to: outputs/results/"
echo ""

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}[3/4] Generating Report Visualizations${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "Creating publication-ready figures..."
echo ""

python src/report_visualization.py || {
    echo -e "${YELLOW}⚠️  Report visualization failed. Check if results exist.${NC}"
    echo -e "${YELLOW}   You may need to run src/main.py first.${NC}"
}

echo -e "${GREEN}✓ Report visualizations complete${NC}"
echo -e "  Figures saved to: outputs/report/"
echo ""

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}[4/4] Summary of Results${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Print summary if files exist
if [ -f "outputs/comparison/model_comparison.csv" ]; then
    echo -e "${GREEN}Model Comparison Results:${NC}"
    cat outputs/comparison/model_comparison.csv | column -t -s,
    echo ""
fi

if [ -f "outputs/results/test_metrics.csv" ]; then
    echo -e "${GREEN}Semi-Supervised Test Results:${NC}"
    cat outputs/results/test_metrics.csv | column -t -s,
    echo ""
fi

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓ ANALYSIS COMPLETE!${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${GREEN}Results saved to:${NC}"
echo "  📊 outputs/comparison/    - Model comparison results"
echo "  📈 outputs/results/       - Semi-supervised learning results"
echo "  📉 outputs/report/        - Publication-ready figures"
echo ""
echo -e "${GREEN}Key files for your report:${NC}"
echo "  🎨 outputs/report/multipanel_results.png"
echo "  📊 outputs/comparison/tsne_annotated_*.png"
echo "  📈 outputs/comparison/metrics_comparison.png"
echo "  📋 outputs/report/latex_tables.tex"
echo "  📝 outputs/report/executive_summary.txt"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Review results in outputs/ directory"
echo "  2. Check latex_tables.tex for report tables"
echo "  3. Use figures in outputs/report/ for your report"
echo "  4. See REPORT_GENERATION_GUIDE.md for more details"
echo ""
