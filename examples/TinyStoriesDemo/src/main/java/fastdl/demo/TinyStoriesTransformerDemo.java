package fastdl.demo;

import fastdl.data.TextDataset;
import fastdl.generation.Generator;
import fastdl.model.GPTConfig;
import fastdl.model.GPTModel;
import fastdl.optim.AdamW;
import fastdl.tokenizer.CharTokenizer;
import fastdl.training.Trainer;
import fastdl.training.TrainerConfig;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * FastDL — TinyStories Transformer Demo
 *
 * Real Transformer (GPT-style) trained on TinyStories text.
 * Uses CharTokenizer + GPTModel + Trainer + Generator.
 *
 * Run via: run-tiny2.bat
 */
public class TinyStoriesTransformerDemo {

    // ANSI palette (grau/weiss style — FastAIBot/FastAIRag pattern)
    private static final String RESET  = "\u001B[0m";
    private static final String BOLD   = "\u001B[1m";
    private static final String FG     = "\u001B[38;5;252m";
    private static final String MUTED  = "\u001B[38;5;245m";
    private static final String ACCENT = "\u001B[38;5;208m";
    private static final String GOOD   = "\u001B[38;5;114m";
    private static final String WARN   = "\u001B[38;5;221m";
    private static final String BAR    = "\u001B[38;5;240m";

    // Demo limits — keep it fast on CPU
    private static final int MAX_CHARS  = 200_000;  // ~200k chars from corpus
    private static final int SEQ_LEN    = 64;
    private static final int BATCH_SIZE = 2;
    private static final int MAX_ITER   = 150;
    private static final int EVAL_EVERY = 25;
    private static final int GEN_TOKENS = 120;

    public static void main(String[] args) throws Exception {
        printHeader();

        // ---- 1. Locate dataset ----
        String dataPath = resolveDatasetPath();
        if (dataPath == null) {
            System.out.println(MUTED + "No TinyStories dataset found." + RESET);
            System.out.println(FG + "Download TinyStories-train.txt from Hugging Face" + RESET);
            System.out.println(MUTED + "and place it under: data/TinyStories-train.txt" + RESET);
            return;
        }
        System.out.println(ACCENT + "Dataset : " + RESET + dataPath);

        // ---- 2. Build tokenizer from corpus sample ----
        System.out.println(FG + "Building tokenizer..." + RESET);
        String corpus = readSample(dataPath, MAX_CHARS);
        CharTokenizer tokenizer = CharTokenizer.build(corpus);
        System.out.println(GOOD + "Tokenizer: " + RESET + tokenizer);

        // ---- 3. Build dataset ----
        System.out.println(FG + "Building dataset..." + RESET);
        TextDataset dataset;
        try {
            dataset = new TextDataset(dataPath, SEQ_LEN, tokenizer, MAX_CHARS);
        } catch (IOException e) {
            System.err.println("[ERROR] " + e.getMessage());
            return;
        }

        // ---- 4. Build model ----
        GPTConfig config = new GPTConfig(
            tokenizer.vocabSize(), SEQ_LEN,
            /*dModel*/   128,
            /*numHeads*/ 4,
            /*numLayers*/4,
            /*dropout*/  0.0f
        );
        GPTModel model = new GPTModel(config);
        System.out.println(GOOD + "Model   : " + RESET + model);
        System.out.println(GOOD + "Params  : " + RESET + String.format("%,d", model.paramCount()));

        // ---- 5. Optimizer ----
        AdamW optimizer = new AdamW(model.parameters(), 3e-4f, 0.01f);

        // ---- 6. Trainer ----
        TrainerConfig tConfig = new TrainerConfig(
            BATCH_SIZE, MAX_ITER, EVAL_EVERY, 4,
            3e-4f, 0.01f, 1.0f, null
        );
        Trainer trainer = new Trainer(model, dataset, optimizer, tConfig);

        System.out.println();
        System.out.println(BOLD + "Training" + RESET
            + MUTED + "  " + MAX_ITER + " steps | batch=" + BATCH_SIZE
            + " | seqLen=" + SEQ_LEN + RESET);
        System.out.println(BAR + "────────────────────────────────────────────────────" + RESET);

        long startMs = System.currentTimeMillis();

        trainer.setOnStep(info -> {
            long elapsed = System.currentTimeMillis() - startMs;
            String bar = progressBar(info.progressPct(), 20);
            System.out.printf(ACCENT + "[%4d/%d]" + RESET
                + " " + FG + "train=%.4f" + RESET
                + "  " + WARN + "eval=%.4f" + RESET
                + "  " + BAR + "%s" + RESET
                + "  " + MUTED + "%ds" + RESET + "%n",
                info.step, info.totalSteps,
                info.trainLoss, info.evalLoss,
                bar,
                elapsed / 1000);
        });

        trainer.train();

        long elapsed = System.currentTimeMillis() - startMs;
        System.out.println(BAR + "────────────────────────────────────────────────────" + RESET);
        System.out.println(GOOD + "Training complete" + RESET
            + MUTED + "  " + (elapsed / 1000) + "s elapsed" + RESET);

        // ---- 7. Generation ----
        Generator generator = new Generator(model, tokenizer);

        System.out.println();
        System.out.println(BOLD + "Generation" + RESET);
        System.out.println(BAR + "────────────────────────────────────────────────────" + RESET);

        String[] prompts = pickPrompts(corpus, 3);
        for (String prompt : prompts) {
            System.out.println(MUTED + "Prompt  : " + RESET + prompt);

            // greedy
            String greedy = generator.generate(prompt, GEN_TOKENS, 1.0f, 1);
            System.out.println(ACCENT + "Greedy  : " + RESET + clip(greedy, 200));

            // temperature 0.8
            String sampled = generator.generate(prompt, GEN_TOKENS, 0.8f, 40);
            System.out.println(GOOD  + "Sampled : " + RESET + clip(sampled, 200));

            System.out.println();
        }
    }

    // -------------------------------------------------------------------------

    private static void printHeader() {
        System.out.println(FG + "════════════════════════════════════════════════════" + RESET);
        System.out.println(BOLD + " FastDL  —  TinyStories Transformer Demo" + RESET);
        System.out.println(MUTED + " GPT-style causal language model trained from scratch" + RESET);
        System.out.println(FG + "════════════════════════════════════════════════════" + RESET);
        System.out.println();
    }

    private static String resolveDatasetPath() {
        List<String> candidates = List.of(
            "data/TinyStories-train.txt",
            "data/tinystories-train.txt",
            "data/tinystories.txt",
            "data/TinyStories.txt",
            "data/train.txt",
            "TinyStories-train.txt"
        );
        Path cwd = Path.of("").toAbsolutePath();
        for (int depth = 0; depth < 6; depth++) {
            for (String c : candidates) {
                Path p = cwd.resolve(c).normalize();
                if (Files.isRegularFile(p)) return p.toString();
            }
            Path parent = cwd.getParent();
            if (parent == null) break;
            cwd = parent;
        }
        return null;
    }

    private static String readSample(String path, int maxChars) throws IOException {
        String raw = Files.readString(Path.of(path));
        return maxChars > 0 && raw.length() > maxChars ? raw.substring(0, maxChars) : raw;
    }

    private static String[] pickPrompts(String corpus, int count) {
        // pick short sentences from corpus as generation seeds
        String[] sentences = corpus.split("[.!?]");
        String[] out = new String[count];
        int picked = 0;
        for (String s : sentences) {
            String clean = s.strip();
            if (clean.length() >= 10 && clean.length() <= 40) {
                out[picked++] = clean.substring(0, Math.min(clean.length(), 20));
                if (picked == count) break;
            }
        }
        while (picked < count) out[picked++] = "Once upon a time";
        return out;
    }

    private static String progressBar(float pct, int width) {
        int filled = (int) (pct / 100f * width);
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < width; i++) sb.append(i < filled ? '█' : '░');
        sb.append(']');
        return sb.toString();
    }

    private static String clip(String s, int max) {
        return s.length() > max ? s.substring(0, max) + "…" : s;
    }
}
