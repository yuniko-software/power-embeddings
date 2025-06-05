# BGE-M3 ONNX Research

This directory contains research and implementation for converting the BGE-M3 multilingual embedding model to ONNX format, enabling cross-platform usage with identical results to the original FlagEmbedding implementation.

## Contents

- **`bge-m3-to-onnx.ipynb`** - Jupyter notebook documenting the conversion process from FlagEmbedding to ONNX
- **`generate_reference_embeddings.py`** - Python script to generate reference embeddings for testing
- **`dotnet/`** - C# implementation using the ONNX models
- **`onnx/`** - Directory containing the ONNX model files (not included in git)
- **`run_tests.sh`** - Bash script to run the complete test suite
- **`run_tests.ps1`** - PowerShell script to run the complete test suite

## Setup

### Step 1: Generate ONNX Models
1. Open and run `bge-m3-to-onnx.ipynb` in Jupyter
2. This will create the required ONNX files in the `onnx/` directory
3. The notebook downloads the BGE-M3 model and converts it to ONNX format

### Step 2: Run test scripts

#### Windows
```powershell
./run_tests.ps1
```

#### Linux/macOS
```bash
chmod +x run_tests.sh
./run_tests.sh
```

## What the test scripts do

1. **Verify** ONNX models are present
2. **Install** required Python packages
3. **Generate** reference embeddings using Python/ONNX
4. **Run** C# tests to validate cross-language consistency
5. **Report** success/failure with clear output

## Manual Usage

### Generate Reference Embeddings
```bash
python3 generate_reference_embeddings.py
```

### Run C# Tests
```bash
cd dotnet/PowerEmbeddings.Research.BgeM3.Onnx.Tests
dotnet test
```

### Use C# Implementation
```csharp
using var embedder = new M3Embedder(tokenizerPath, modelPath);
var result = embedder.GenerateEmbeddings("Your text here");

// Access all three embedding types:
var dense = result.DenseEmbedding;      // Dense vectors
var sparse = result.SparseWeights;     // Sparse weights
var colbert = result.ColBertVectors;   // ColBERT vectors
```