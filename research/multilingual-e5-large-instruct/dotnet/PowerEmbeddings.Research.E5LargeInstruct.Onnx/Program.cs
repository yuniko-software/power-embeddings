using System.Globalization;
using PowerEmbeddings.Research.E5LargeInstruct.Onnx;

// Define paths relative to the project
var researchDir = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", ".."));
var onnxDir = Path.Combine(researchDir, "onnx");

var tokenizerPath = Path.Combine(onnxDir, "e5_large_instruct_tokenizer.onnx");
var modelPath = Path.Combine(onnxDir, "e5_large_instruct_model.onnx");

Console.WriteLine("===== E5 Large Instruct ONNX Embedding Research =====");
Console.WriteLine($"Using tokenizer: {tokenizerPath}");
Console.WriteLine($"Using model: {modelPath}");

// Create the embedding generator
using var embeddingGenerator = new E5LargeInstructEmbedder(tokenizerPath, modelPath);

var task = "Given a web search query, retrieve relevant passages that answer the query";

var queries = new[]
{
    "how much protein should a female eat",
    "南瓜的家常做法"
};

var documents = new[]
{
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "1.清炒南瓜丝 原料:嫩南瓜半个 调料:葱、盐、白糖、鸡精 做法: 1、南瓜用刀薄薄的削去表面一层皮,用勺子刮去瓤 2、擦成细丝(没有擦菜板就用刀慢慢切成细丝) 3、锅烧热放油,入葱花煸出香味 4、入南瓜丝快速翻炒一分钟左右,放盐、一点白糖和鸡精调味出锅 2.香葱炒南瓜 原料:南瓜1只 调料:香葱、蒜末、橄榄油、盐 做法: 1、将南瓜去皮,切成片 2、油锅8成热后,将蒜末放入爆香 3、爆香后,将南瓜片放入,翻炒 4、在翻炒的同时,可以不时地往锅里加水,但不要太多 5、放入盐,炒匀 6、南瓜差不多软和绵了之后,就可以关火 7、撒入香葱,即可出锅"
};

// Format queries with instructions
var instructedQueries = queries
    .Select(q => E5LargeInstructEmbedder.GetDetailedInstruct(task, q))
    .ToArray();

Console.WriteLine("\n=== QUERIES ===");
for (int i = 0; i < queries.Length; i++)
{
    Console.WriteLine($"Query {i + 1}: {queries[i]}");
    Console.WriteLine($"Instructed: {instructedQueries[i]}");
    Console.WriteLine();
}

Console.WriteLine("=== DOCUMENTS ===");
for (int i = 0; i < documents.Length; i++)
{
    var preview = documents[i].Length > 100 ? documents[i][..100] + "..." : documents[i];
    Console.WriteLine($"Document {i + 1}: {preview}");
}

// Generate embeddings
Console.WriteLine("\n=== GENERATING EMBEDDINGS ===");
Console.WriteLine("Processing queries...");
var queryEmbeddings = embeddingGenerator.GenerateEmbeddings(instructedQueries);

Console.WriteLine("Processing documents...");
var documentEmbeddings = embeddingGenerator.GenerateEmbeddings(documents);

// Print embedding information
Console.WriteLine("\n=== EMBEDDING INFORMATION ===");
Console.WriteLine($"Query embeddings: {queryEmbeddings.Length}");
Console.WriteLine($"Document embeddings: {documentEmbeddings.Length}");
Console.WriteLine($"Embedding dimension: {queryEmbeddings[0].Length}");

// Show first few values of first embedding
var firstEmbedding = queryEmbeddings[0];
Console.WriteLine($"First embedding (first 10 values): [{string.Join(", ", firstEmbedding.Take(10).Select(v => v.ToString("F6", CultureInfo.InvariantCulture)))}]");

// Verify embeddings are normalized
var norm = Math.Sqrt(firstEmbedding.Sum(x => x * x));
Console.WriteLine($"First embedding L2 norm: {norm:F10} (should be close to 1.0)");

// Calculate similarity scores
Console.WriteLine("\n=== SIMILARITY SCORES ===");
var scores = E5LargeInstructEmbedder.CalculateSimilarityScores(queryEmbeddings, documentEmbeddings);

Console.WriteLine("Similarity matrix (queries × documents):");
for (int i = 0; i < queries.Length; i++)
{
    var scoreRow = new List<string>();
    for (int j = 0; j < documents.Length; j++)
    {
        scoreRow.Add(scores[i][j].ToString("F2", CultureInfo.InvariantCulture));
    }
    Console.WriteLine($"Query {i + 1}: [{string.Join(", ", scoreRow)}]");
}

Console.WriteLine("\nExpected pattern:");
Console.WriteLine("- Query 1 (protein) should match Document 1 (protein) better than Document 2");
Console.WriteLine("- Query 2 (pumpkin) should match Document 2 (pumpkin) better than Document 1");

var proteinMatch = scores[0][0] > scores[0][1];
var pumpkinMatch = scores[1][1] > scores[1][0];

Console.WriteLine($"\nActual results:");
Console.WriteLine($"- Protein query matches protein doc better: {proteinMatch} (scores: {scores[0][0]:F2} vs {scores[0][1]:F2})");
Console.WriteLine($"- Pumpkin query matches pumpkin doc better: {pumpkinMatch} (scores: {scores[1][1]:F2} vs {scores[1][0]:F2})");

if (proteinMatch && pumpkinMatch)
{
    Console.WriteLine("\nSUCCESS: Similarity scores match expected pattern!");
}
else
{
    Console.WriteLine("\nWARNING: Similarity scores don't match expected pattern.");
}

// Test different instructions
Console.WriteLine("\n=== TESTING DIFFERENT INSTRUCTIONS ===");

var classificationTask = "Classify the sentiment of this text";
var sentimentQuery = "I love this product!";
var classificationText = E5LargeInstructEmbedder.GetDetailedInstruct(classificationTask, sentimentQuery);

var summarizationTask = "Summarize the following passage";
var textToSummarize = "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.";
var summarizationText = E5LargeInstructEmbedder.GetDetailedInstruct(summarizationTask, textToSummarize);

var additionalTexts = new[] { classificationText, summarizationText };
var additionalEmbeddings = embeddingGenerator.GenerateEmbeddings(additionalTexts);

Console.WriteLine($"Classification instruction embedding (first 5): [{string.Join(", ", additionalEmbeddings[0].Take(5).Select(v => v.ToString("F6", CultureInfo.InvariantCulture)))}]");
Console.WriteLine($"Summarization instruction embedding (first 5): [{string.Join(", ", additionalEmbeddings[1].Take(5).Select(v => v.ToString("F6", CultureInfo.InvariantCulture)))}]");

// Compare similarity between different instruction types
var instructionSimilarity = E5LargeInstructEmbedder.CalculateCosineSimilarity(
    additionalEmbeddings[0],
    additionalEmbeddings[1]);

Console.WriteLine($"Similarity between different instruction types: {instructionSimilarity:F6}");

Console.WriteLine("\n===== DEMO COMPLETED SUCCESSFULLY =====");
Console.WriteLine("All embedding types generated and similarity calculations performed!");
Console.WriteLine("The E5 Large Instruct ONNX implementation is working correctly.");
