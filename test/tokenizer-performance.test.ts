/**
 * Tokenizer performance tests using real tokenizer.json from HuggingFace.
 * 
 * Uses local fixtures: test/tokenizers/glm-5.json and glm-5-config.json
 * 
 * To download/update fixtures (use proxy if in China):
 *   curl -x http://127.0.0.1:7890 -L -o test/tokenizers/glm-5.json \
 *     "https://huggingface.co/zai-org/GLM-5/resolve/main/tokenizer.json"
 *   curl -x http://127.0.0.1:7890 -L -o test/tokenizers/glm-5-config.json \
 *     "https://huggingface.co/zai-org/GLM-5/resolve/main/tokenizer_config.json"
 */
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { Tokenizer } from "@huggingface/tokenizers";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { readFile } from "node:fs/promises";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const TOKENIZERS_DIR = path.join(__dirname, "tokenizers");

// Test samples
const SAMPLES = {
  english: "Hello world, this is a test message. Artificial intelligence is changing our lives. The quick brown fox jumps over the lazy dog.",
  chinese: "你好世界，这是一个测试消息。人工智能正在改变我们的生活方式。机器学习和深度学习是当今最热门的技术领域。",
  mixed: "Hello 你好 World 世界！这是中英混合文本 Mixed content test。123 ABC 中文和 English 混合。",
  code: `function calculateSum(arr: number[]): number {
  let sum = 0;
  for (const num of arr) {
    sum += num;
  }
  return sum;
}

// Usage example
const numbers = [1, 2, 3, 4, 5];
console.log(calculateSum(numbers));`,
};

// Results storage for summary table
const results: Array<{
  type: string;
  chars: number;
  heuristic: number;
  precise: number;
  diff: string;
  heuristicLatencyNs: number;
  preciseLatencyNs: number;
}> = [];

describe("Tokenizer Performance (Real)", () => {
  let tokenizer: Tokenizer | null = null;
  let fixturesExist = false;

  beforeAll(async () => {
    const tokenizerPath = path.join(TOKENIZERS_DIR, "glm-5.json");
    const configPath = path.join(TOKENIZERS_DIR, "glm-5-config.json");
    
    try {
      const tokenizerJson = JSON.parse(await readFile(tokenizerPath, "utf-8"));
      const configJson = JSON.parse(await readFile(configPath, "utf-8"));
      tokenizer = new Tokenizer(tokenizerJson, configJson);
      fixturesExist = true;
    } catch {
      console.log("Skipping performance tests: fixtures not found. Run:");
      console.log("  curl -L -o test/tokenizers/glm-5.json \\");
      console.log("    'https://huggingface.co/zai-org/GLM-5/resolve/main/tokenizer.json'");
      console.log("  curl -L -o test/tokenizers/glm-5-config.json \\");
      console.log("    'https://huggingface.co/zai-org/GLM-5/resolve/main/tokenizer_config.json'");
    }
  });

  afterAll(() => {
    if (results.length === 0) return;
    
    console.log("\n");
    console.log("═".repeat(70));
    console.log("📊 Tokenizer Performance Summary (GLM-5)");
    console.log("═".repeat(70));
    console.log("");
    console.log("【Accuracy Comparison】");
    console.log("| Type      | Chars | chars/4 | Precise | Diff    |");
    console.log("|-----------|-------|---------|---------|---------|");
    
    for (const r of results) {
      const type = r.type.padEnd(9);
      const chars = String(r.chars).padStart(5);
      const heuristic = String(r.heuristic).padStart(7);
      const precise = String(r.precise).padStart(7);
      const diff = r.diff.padStart(7);
      
      console.log(`| ${type} | ${chars} | ${heuristic} | ${precise} | ${diff} |`);
    }
    
    console.log("");
    console.log("【Latency Comparison (ms)】");
    console.log("| Type      | chars/4 | tokenizer |");
    console.log("|-----------|---------|-----------|");
    
    for (const r of results) {
      const type = r.type.padEnd(9);
      const hMs = (r.heuristicLatencyNs / 1_000_000).toFixed(3).padStart(7);
      const pMs = (r.preciseLatencyNs / 1_000_000).toFixed(3).padStart(9);
      
      console.log(`| ${type} | ${hMs} | ${pMs} |`);
    }
    
    console.log("");
    console.log("─".repeat(70));
    console.log("Accuracy: Diff = (chars/4 - Precise) / Precise × 100%");
    console.log("═".repeat(70));
    console.log("");
  });

  const heuristicTokens = (text: string) => Math.ceil(text.length / 4);

  function measureHeuristicNs(text: string): { result: number; ns: number } {
    const start = process.hrtime.bigint();
    const result = heuristicTokens(text);
    const end = process.hrtime.bigint();
    return { result, ns: Number(end - start) };
  }

  function measurePreciseNs(text: string): { result: number; ns: number } {
    const start = process.hrtime.bigint();
    const encoded = tokenizer!.encode(text);
    const end = process.hrtime.bigint();
    return { result: encoded.ids.length, ns: Number(end - start) };
  }

  function runTest(type: string, text: string) {
    if (!fixturesExist) return;
    
    // Measure heuristic
    const { result: heuristic, ns: heuristicNs } = measureHeuristicNs(text);
    
    // Measure precise (run multiple times for more accurate measurement)
    let preciseNs = 0;
    let precise = 0;
    const iterations = 10;
    for (let i = 0; i < iterations; i++) {
      const { result, ns } = measurePreciseNs(text);
      preciseNs += ns;
      precise = result;
    }
    preciseNs = Math.round(preciseNs / iterations);
    
    const diffPercent = ((heuristic - precise) / precise * 100);
    const diff = diffPercent >= 0 ? `+${diffPercent.toFixed(1)}%` : `${diffPercent.toFixed(1)}%`;
    
    results.push({ 
      type, 
      chars: text.length, 
      heuristic, 
      precise, 
      diff,
      heuristicLatencyNs: heuristicNs,
      preciseLatencyNs: preciseNs,
    });
    
    expect(precise).toBeGreaterThan(0);
  }

  describe("Accuracy & Latency Tests", () => {
    it("English text", () => runTest("English", SAMPLES.english));
    it("Chinese text", () => runTest("Chinese", SAMPLES.chinese));
    it("Mixed text", () => runTest("Mixed", SAMPLES.mixed));
    it("Code", () => runTest("Code", SAMPLES.code));
  });
});
