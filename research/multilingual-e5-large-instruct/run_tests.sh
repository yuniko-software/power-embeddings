#!/bin/bash
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting E5 Large Instruct ONNX model tests${NC}"

# Verify ONNX models exist (script is already in research/e5-large-instruct)
if [ ! -d "onnx" ]; then
    echo -e "${RED}ERROR: onnx directory not found!${NC}"
    echo "Please create an 'onnx' directory in research/e5-large-instruct and add the required ONNX files."
    exit 1
fi

if [ ! -f "onnx/e5_large_instruct_tokenizer.onnx" ]; then
    echo -e "${RED}ERROR: e5_large_instruct_tokenizer.onnx not found!${NC}"
    echo "Please download or generate the E5 Large Instruct tokenizer ONNX model and place it in the onnx directory."
    exit 1
fi

if [ ! -f "onnx/e5_large_instruct_model.onnx" ]; then
    echo -e "${RED}ERROR: e5_large_instruct_model.onnx not found!${NC}"
    echo "Please download the E5 Large Instruct model ONNX file and place it in the onnx directory."
    exit 1
fi

# Step 1: Generate reference embeddings using Python
echo -e "${YELLOW}Generating reference embeddings using Python...${NC}"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: python3 command not found!${NC}"
    echo "Please install Python 3 to run this test script."
    exit 1
fi

echo "Found Python: $(python3 --version)"

# Check if required packages are installed
PACKAGES=("onnxruntime" "onnxruntime-extensions" "numpy")
MISSING_PACKAGES=()

for pkg in "${PACKAGES[@]}"; do
    # Convert dashes to underscores for import check
    pkg_import=${pkg//-/_}
    if ! python3 -c "import $pkg_import" &> /dev/null; then
        MISSING_PACKAGES+=("$pkg")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo -e "${YELLOW}Installing required Python packages: ${MISSING_PACKAGES[*]}${NC}"
    pip install "${MISSING_PACKAGES[@]}"
    if [ $? -ne 0 ]; then
        echo -e "${RED}ERROR: Failed to install required packages${NC}"
        exit 1
    fi
fi

# Run the Python script to generate reference embeddings
python3 generate_reference_embeddings.py
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to generate reference embeddings!${NC}"
    exit 1
fi

echo -e "${GREEN}Reference embeddings generated successfully!${NC}"

# Step 2: Run .NET tests
echo -e "${YELLOW}Running .NET tests...${NC}"

pushd dotnet/PowerEmbeddings.Research.E5LargeInstruct.Onnx.Tests > /dev/null
dotnet test --verbosity normal

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: .NET tests failed!${NC}"
    popd > /dev/null
    exit 1
fi
popd > /dev/null

echo -e "${GREEN}.NET tests passed successfully!${NC}"
echo -e "${GREEN}All E5 Large Instruct tests passed successfully!${NC}"