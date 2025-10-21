#!/bin/bash
#
# Fine-Tuning Quick Reference Script
# This script provides common fine-tuning commands for easy copy-paste
#

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}HATE SPEECH MODEL FINE-TUNING COMMANDS${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

echo -e "${GREEN}1. Basic Fine-Tuning (Default Settings)${NC}"
echo "   python src/fine_tune_model.py"
echo ""

echo -e "${GREEN}2. Optimized for Hate Speech Detection (Balanced Classes)${NC}"
echo "   python src/fine_tune_model.py --balance-classes --epochs 6"
echo ""

echo -e "${GREEN}3. Quick Test (2 epochs, small batch)${NC}"
echo "   python src/fine_tune_model.py --epochs 2 --batch-size 8"
echo ""

echo -e "${GREEN}4. Full Training with Comparison${NC}"
echo "   python src/fine_tune_model.py --epochs 6 --batch-size 16 --balance-classes --compare"
echo ""

echo -e "${GREEN}5. Evaluate Existing Fine-Tuned Model${NC}"
echo "   python src/fine_tune_model.py --evaluate-only models/fine_tuned --compare"
echo ""

echo -e "${GREEN}6. Run Complete Example${NC}"
echo "   python example_fine_tuning.py"
echo ""

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}INTEGRATION WITH PIPELINE${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

echo -e "${GREEN}1. Update config.yaml to use fine-tuned model${NC}"
echo "   # Edit config.yaml:"
echo "   # embedding:"
echo "   #   model: \"models/fine_tuned\""
echo ""

echo -e "${GREEN}2. Generate embeddings with fine-tuned model${NC}"
echo "   python src/embeddings.py --model models/fine_tuned"
echo ""

echo -e "${GREEN}3. Run main pipeline with fine-tuned embeddings${NC}"
echo "   python src/main.py --config config.yaml"
echo ""

echo -e "${GREEN}4. Compare models using embedding comparison tool${NC}"
echo "   python src/embedding_comparison.py --models all-MiniLM-L6-v2 models/fine_tuned --n-clusters 12"
echo ""

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}HYPERPARAMETER TUNING${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

echo -e "${GREEN}Grid Search Example${NC}"
echo "   for lr in 1e-5 2e-5 3e-5; do"
echo "     for epochs in 4 6; do"
echo "       python src/fine_tune_model.py \\"
echo "         --learning-rate \$lr \\"
echo "         --epochs \$epochs \\"
echo "         --output models/ft_lr\${lr}_e\${epochs}"
echo "     done"
echo "   done"
echo ""

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}TROUBLESHOOTING${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

echo -e "${YELLOW}Out of Memory Error?${NC}"
echo "   python src/fine_tune_model.py --batch-size 8"
echo ""

echo -e "${YELLOW}Use CPU instead of GPU?${NC}"
echo "   CUDA_VISIBLE_DEVICES=\"\" python src/fine_tune_model.py"
echo ""

echo -e "${YELLOW}Check if CUDA is available?${NC}"
echo "   python -c \"import torch; print('CUDA available:', torch.cuda.is_available())\""
echo ""

echo -e "${BLUE}========================================${NC}"
echo ""
echo "For detailed documentation, see: FINE_TUNING_README.md"
echo ""
