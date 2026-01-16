#!/bin/bash

cd "$(dirname "$0")"

echo "=================================="
echo "KPMP Metadata Validation"
echo "=================================="
echo ""

if [ ! -f "data/kpmp.h5ad" ]; then
    echo "❌ Error: data/kpmp.h5ad not found"
    exit 1
fi

echo "✓ Data file exists: data/kpmp.h5ad"
echo ""

python3 explore_kpmp_metadata.py

echo ""
echo "=================================="
echo "Expected Values for BioRSP Script"
echo "=================================="
echo ""
echo "Default parameters (no flags needed):"
echo "  --celltype_key subclass.l1"
echo "  --donor_key donor_id"
echo ""
echo "Disease values (case-insensitive shortcuts work):"
echo "  Healthy_living_donor  →  use 'normal' or 'healthy'"
echo "  AKI                   →  use 'aki' or 'acute'"
echo "  CKD                   →  use 'ckd' or 'chronic'"
echo ""
echo "Cell type filter for TAL:"
echo "  --celltype_filter TAL"
echo ""
