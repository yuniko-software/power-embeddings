using System.Text.Json;
using Xunit;

namespace PowerEmbeddings.Research.E5LargeInstruct.Onnx.Tests;

/// <summary>
/// Reference embedding data structure from Python-generated JSON
/// </summary>
public record E5ReferenceEmbedding(
    string Text,
    float[] Embedding
);

public sealed class E5LargeInstructEmbeddingComparisonTests : IDisposable
{
    private readonly E5LargeInstructEmbedder _embedder;
    private readonly Dictionary<string, E5ReferenceEmbedding> _referenceEmbeddings;

    public E5LargeInstructEmbeddingComparisonTests()
    {
        var researchE5Dir = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", ".."));
        var onnxDir = Path.Combine(researchE5Dir, "onnx");

        var tokenizerPath = Path.Combine(onnxDir, "e5_large_instruct_tokenizer.onnx");
        var modelPath = Path.Combine(onnxDir, "e5_large_instruct_model.onnx");
        var referenceFile = Path.Combine(onnxDir, "e5_large_instruct_reference_embeddings.json");

        if (!File.Exists(tokenizerPath))
        {
            throw new FileNotFoundException($"Tokenizer file not found at {tokenizerPath}");
        }
        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException($"Model file not found at {modelPath}");
        }
        if (!File.Exists(referenceFile))
        {
            throw new FileNotFoundException($"Reference embeddings file not found at {referenceFile}. Please run the Python script to generate reference embeddings first.");
        }

        _embedder = new E5LargeInstructEmbedder(tokenizerPath, modelPath);

        var jsonContent = File.ReadAllText(referenceFile);
        var rawEmbeddings = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(jsonContent)!;

        _referenceEmbeddings = [];

        foreach (var kvp in rawEmbeddings)
        {
            var element = kvp.Value;

            var text = element.GetProperty("text").GetString()!;
            var embedding = element.GetProperty("embedding").EnumerateArray()
                .Select(x => (float)x.GetDouble()).ToArray();

            _referenceEmbeddings[kvp.Key] = new E5ReferenceEmbedding(
                Text: text,
                Embedding: embedding
            );
        }
    }

    [Fact]
    public void AllEmbeddings_ShouldMatchPythonEmbeddings()
    {
        var failedComparisons = new List<string>();

        foreach (var entry in _referenceEmbeddings)
        {
            var testName = entry.Key;
            var referenceEmbedding = entry.Value;

            try
            {
                var result = _embedder.GenerateEmbedding(referenceEmbedding.Text);

                var similarity = E5LargeInstructEmbedder.CalculateCosineSimilarity(
                    result, referenceEmbedding.Embedding);

                if (similarity <= 0.9999)
                {
                    failedComparisons.Add($"Similarity {similarity:F10} for '{testName}'");
                }
            }
            catch (Exception ex)
            {
                failedComparisons.Add($"Exception for '{testName}': {ex.Message}");
            }
        }

        if (failedComparisons.Any())
        {
            var errorMessage = $"Embedding comparison failures:\n{string.Join("\n", failedComparisons)}";
            Assert.Fail(errorMessage);
        }
    }

    public void Dispose()
    {
        _embedder?.Dispose();
    }
}