using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace PowerEmbeddings.Research.E5LargeInstruct.Onnx;

/// <summary>
/// Provides functionality to generate embeddings using ONNX E5 Large Instruct model
/// </summary>
public class E5LargeInstructEmbedder : IDisposable
{
    private readonly InferenceSession _tokenizerSession;
    private readonly InferenceSession _modelSession;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the E5LargeInstructEmbedder class
    /// </summary>
    /// <param name="tokenizerPath">Path to the ONNX tokenizer model</param>
    /// <param name="modelPath">Path to the ONNX embedding model</param>
    public E5LargeInstructEmbedder(string tokenizerPath, string modelPath)
    {
        using var tokenizerOptions = new SessionOptions();
        tokenizerOptions.RegisterOrtExtensions();
        _tokenizerSession = new InferenceSession(tokenizerPath, tokenizerOptions);

        _modelSession = new InferenceSession(modelPath);
    }

    /// <summary>
    /// Formats a query with instruction as required by E5 Large Instruct
    /// </summary>
    /// <param name="taskDescription">Description of the task</param>
    /// <param name="query">The actual query</param>
    /// <returns>Formatted instruction + query string</returns>
    public static string GetDetailedInstruct(string taskDescription, string query)
    {
        return $"Instruct: {taskDescription}\nQuery: {query}";
    }

    /// <summary>
    /// Generates embeddings for a single text
    /// </summary>
    /// <param name="text">The input text</param>
    /// <returns>Normalized embedding vector</returns>
    public float[] GenerateEmbedding(string text)
    {
        return GenerateEmbeddings([text])[0];
    }

    /// <summary>
    /// Generates embeddings for multiple texts
    /// </summary>
    /// <param name="texts">The input texts</param>
    /// <returns>Array of normalized embedding vectors</returns>
    public float[][] GenerateEmbeddings(string[] texts)
    {
        var results = new float[texts.Length][];

        for (int i = 0; i < texts.Length; i++)
        {
            results[i] = ProcessSingleText(texts[i]);
        }

        return results;
    }

    /// <summary>
    /// Calculate cosine similarity between two embedding vectors
    /// </summary>
    /// <param name="a">First embedding vector</param>
    /// <param name="b">Second embedding vector</param>
    /// <returns>Cosine similarity score</returns>
    public static double CalculateCosineSimilarity(float[] a, float[] b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException("Vectors must be of the same length");
        }

        double dotProduct = 0;
        double normA = 0;
        double normB = 0;

        for (int i = 0; i < a.Length; i++)
        {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        return dotProduct / (Math.Sqrt(normA) * Math.Sqrt(normB));
    }

    /// <summary>
    /// Calculate similarity scores between queries and documents
    /// </summary>
    /// <param name="queryEmbeddings">Query embeddings</param>
    /// <param name="documentEmbeddings">Document embeddings</param>
    /// <param name="scale">Scale factor (default 100 to match original example)</param>
    /// <returns>Similarity score matrix as jagged array</returns>
    public static float[][] CalculateSimilarityScores(float[][] queryEmbeddings,
        float[][] documentEmbeddings, float scale = 100f)
    {
        var scores = new float[queryEmbeddings.Length][];

        for (int i = 0; i < queryEmbeddings.Length; i++)
        {
            scores[i] = new float[documentEmbeddings.Length];
            for (int j = 0; j < documentEmbeddings.Length; j++)
            {
                var similarity = CalculateCosineSimilarity(
                    queryEmbeddings[i],
                    documentEmbeddings[j]);
                scores[i][j] = (float)(similarity * scale);
            }
        }

        return scores;
    }

    private float[] ProcessSingleText(string text)
    {
        // Create input tensor for tokenizer
        var stringTensor = new DenseTensor<string>([1]);
        stringTensor[0] = text;

        // Create input for tokenizer
        var tokenizerInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("inputs", stringTensor)
        };

        // Run tokenizer
        using var tokenizerResults = _tokenizerSession.Run(tokenizerInputs);
        var tokenizerResultsList = tokenizerResults.ToList();

        // Extract tokens and token_indices (order: tokens, instance_indices, token_indices)
        var tokens = tokenizerResultsList[0].AsTensor<int>().ToArray();
        var tokenIndices = tokenizerResultsList[2].AsTensor<int>().ToArray();

        // Convert to input_ids by sorting tokens based on token_indices
        var tokenPairs = tokens.Zip(tokenIndices, (t, i) => (token: t, index: i))
            .OrderBy(p => p.index)
            .Select(p => p.token)
            .ToArray();

        // Create input_ids tensor with shape [1, tokenPairs.Length]
        var inputIdsTensor = new DenseTensor<long>([1, tokenPairs.Length]);
        for (int i = 0; i < tokenPairs.Length; i++)
        {
            inputIdsTensor[0, i] = tokenPairs[i];
        }

        // Create attention_mask as all 1s with same shape as input_ids
        var attentionMaskTensor = new DenseTensor<long>([1, tokenPairs.Length]);
        for (int i = 0; i < tokenPairs.Length; i++)
        {
            attentionMaskTensor[0, i] = 1;
        }

        // Run the model with the prepared inputs
        var modelInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor)
        };

        using var modelResults = _modelSession.Run(modelInputs);
        var modelResultsList = modelResults.ToList();

        // Extract normalized embeddings
        var embeddingTensor = modelResultsList[0].AsTensor<float>();
        return [.. embeddingTensor]; // Convert to array and remove batch dimension
    }

    /// <summary>
    /// Disposes the resources used by the E5LargeInstructEmbedder
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                _tokenizerSession?.Dispose();
                _modelSession?.Dispose();
            }

            _disposed = true;
        }
    }

    ~E5LargeInstructEmbedder()
    {
        Dispose(false);
    }
}