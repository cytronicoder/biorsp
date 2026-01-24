#!/bin/bash

set -e

echo "=========================================="
echo "Disease-Stratified BioRSP Analysis Workflow"
echo "=========================================="
echo ""

DATA_FILE="data/kpmp.h5ad"
BASE_DIR="results/example_workflow"
CELLTYPE="TAL"
CELLTYPE_KEY="subclass.l1"

if [ ! -f "$DATA_FILE" ]; then
    echo "ERROR: Data file not found: $DATA_FILE"
    echo "Please download or specify the correct path to your KPMP h5ad file"
    exit 1
fi

echo "Step 1: Quick pilot test (1-2 minutes)"
echo "----------------------------------------"
echo "Testing with 50 genes and 100 permutations..."
echo ""

python analysis/kidney_atlas/run_disease_stratified_analysis.py \
  --ref-data "$DATA_FILE" \
  --outdir "${BASE_DIR}/pilot" \
  --celltype-key "$CELLTYPE_KEY" \
  --celltype-filter "$CELLTYPE" \
  --max-genes 50 \
  --n-permutations 100 \
  --subsample 5000

echo ""
echo "✓ Pilot test complete!"
echo "  Results: ${BASE_DIR}/pilot/"
echo ""

echo "Step 2: Review pilot results"
echo "----------------------------------------"
cat "${BASE_DIR}/pilot/README.md"
echo ""

echo "Step 3: Generate comparison report"
echo "----------------------------------------"
python analysis/kidney_atlas/compare_disease_results.py \
  "${BASE_DIR}/pilot"

echo ""
echo "✓ Comparison complete!"
echo ""

echo "Step 4: Optional - Run full analysis"
echo "----------------------------------------"
echo "To run a more comprehensive analysis with more genes and permutations, use:"
echo ""
echo "python analysis/kidney_atlas/run_disease_stratified_analysis.py \\"
echo "  --ref-data $DATA_FILE \\\"
echo "  --outdir ${BASE_DIR}/full \\\"
echo "  --celltype-key $CELLTYPE_KEY \\\"
echo "  --celltype-filter $CELLTYPE \\\"
echo "  --controls 'SLC12A1,UMOD,EGF' \\\"
echo "  --max-genes 500 \\\"
echo "  --n-permutations 1000 \\\"
echo "  --do-genegene \\\"
echo "  --n-workers 4"
echo ""

read -p "Would you like to run the full analysis now? (y/N) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running full analysis (this may take 30-60 minutes)..."
    echo ""
    
    python analysis/kidney_atlas/run_disease_stratified_analysis.py \
      --ref-data "$DATA_FILE" \
      --outdir "${BASE_DIR}/full" \
      --celltype-key "$CELLTYPE_KEY" \
      --celltype-filter "$CELLTYPE" \
      --controls "SLC12A1,UMOD,EGF" \
      --max-genes 500 \
      --n-permutations 1000 \
      --do-genegene \
      --n-workers 4
    
    echo ""
    echo "✓ Full analysis complete!"
    echo "  Results: ${BASE_DIR}/full/"
    echo ""
    
    echo "Generating full comparison with plots..."
    python analysis/kidney_atlas/compare_disease_results.py \
      "${BASE_DIR}/full" \
      --plot \
      --export "${BASE_DIR}/full/comparison.csv"
    
    echo ""
    echo "✓ Comparison with plots complete!"
    echo "  Plots: ${BASE_DIR}/full/comparison_plots/"
    echo "  CSV: ${BASE_DIR}/full/comparison.csv"
else
    echo ""
    echo "Skipping full analysis."
fi

echo ""
echo "=========================================="
echo "Workflow Complete!"
echo "=========================================="
echo ""
echo "Results are in: ${BASE_DIR}/"
echo ""
echo "Next steps:"
echo "1. Review gene_scores.csv in each disease folder"
echo "2. Check radar plots for visual patterns"
echo "3. Compare results across disease conditions"
echo "4. Load CSVs into Python/R for deeper analysis"
echo ""
echo "For help:"
echo "- Quick reference: analysis/kidney_atlas/QUICK_REFERENCE.md"
echo "- Full guide: analysis/kidney_atlas/DISEASE_STRATIFIED_ANALYSIS.md"
echo ""
