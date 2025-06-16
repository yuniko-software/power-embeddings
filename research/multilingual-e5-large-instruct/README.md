# E5 Large Instruct ONNX Research

This directory contains research and implementation for converting the `intfloat/multilingual-e5-large-instruct` model to ONNX format, enabling cross-platform usage with identical results to the original HuggingFace implementation.

## Contents

- **`e5-large-instruct-to-onnx.ipynb`** - Jupyter notebook documenting the conversion process from HuggingFace to ONNX
- **`generate_reference_embeddings.py`** - Python script to generate reference embeddings for testing
- **`dotnet/`** - C# implementation using the ONNX models
- **`onnx/`** - Directory containing the ONNX model files (not included in git)
- **`run_tests.sh`** - Bash script to run the complete test suite
- **`run_tests.ps1`** - PowerShell script to run the complete test suite

## Setup

### Step 1: Generate ONNX Models
1. Open and run `e5-large-instruct-to-onnx.ipynb` in Jupyter
2. This will create the required ONNX files in the `onnx/` directory
3. The notebook downloads the E5 Large Instruct model and converts it to ONNX format

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
cd dotnet/PowerEmbeddings.Research.E5LargeInstruct.Onnx.Tests
dotnet test
```

### Use C# Implementation
```csharp
using var embedder = new E5LargeInstructEmbedder(tokenizerPath, modelPath);

// Format queries with instructions
var task = "Given a web search query, retrieve relevant passages that answer the query";
var query = E5LargeInstructEmbedder.GetDetailedInstruct(task, "how much protein should a female eat");

// Generate embeddings
var queryEmbedding = embedder.GenerateEmbedding(query);
var documentEmbedding = embedder.GenerateEmbedding("Protein requirements document...");

// Calculate similarity
var similarity = E5LargeInstructEmbedder.CalculateCosineSimilarity(
    queryEmbedding, 
    documentEmbedding);
```