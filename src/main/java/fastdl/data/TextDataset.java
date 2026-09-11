package fastdl.data;

import fastdl.tokenizer.Tokenizer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Text dataset for autoregressive (next-token prediction) language model training.
 *
 * <p>Loads a plain text file from disk, tokenizes it using the provided {@link Tokenizer},
 * and stores the resulting flat token ID array in memory. Exposes the corpus as a
 * collection of overlapping fixed-length windows, each of length {@code seqLen + 1},
 * where the first {@code seqLen} tokens are the input and the last {@code seqLen} tokens
 * (shifted by 1) are the targets.
 *
 * <h2>Next-token prediction setup</h2>
 * <pre>
 *   corpus    : [t0, t1, t2, t3, t4, t5, ...]
 *   input  @0 : [t0, t1, t2, t3]   (seqLen=4)
 *   target @0 : [t1, t2, t3, t4]   (shifted by 1 — the "next" token)
 * </pre>
 * At every position the model must predict the next token given all previous tokens.
 * This is the standard language modelling objective (CLM — Causal Language Modelling).
 *
 * <h2>Batch sampling</h2>
 * {@link #getBatch(int)} samples {@code batchSize} random windows from the corpus
 * with replacement using a fixed seed (42) for reproducibility. This means repeated
 * calls to {@code getBatch()} return different windows, enabling stochastic gradient
 * descent without loading the full corpus into a shuffled index on every epoch.
 *
 * <h2>Memory footprint</h2>
 * The entire tokenized corpus is kept as a single {@code int[]} in heap memory.
 * For TinyStories-train.txt (~1.9 GB text) this results in approximately
 * 1.9 billion int values = ~7.6 GB RAM at 4 bytes per int.
 * Use {@code maxChars} to cap the loaded corpus size during development.
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * CharTokenizer tok = CharTokenizer.build(corpus);
 * TextDataset dataset = new TextDataset(
 *     "data/TinyStories-train.txt",
 *     seqLen  = 128,
 *     tok,
 *     maxChars = 500_000    // load only first 500k chars for quick tests
 * );
 *
 * TextDataset.Batch batch = dataset.getBatch(batchSize=4);
 * // batch.inputs  [4][128]  — input token IDs
 * // batch.targets [4][128]  — target token IDs (shifted)
 * }</pre>
 *
 * @see fastdl.tokenizer.Tokenizer
 * @see fastdl.tokenizer.CharTokenizer
 * @see fastdl.training.Trainer
 */
public class TextDataset {

    private final int seqLen;
    private final int[] tokens;          // full tokenized corpus
    private final List<Integer> starts;  // start indices of all valid windows
    private final Random rng;

    /**
     * Load a plain text file and tokenize it.
     *
     * @param path      path to text file
     * @param seqLen    context window length (tokens per sample)
     * @param tokenizer tokenizer to use
     * @param maxChars  max characters to read (0 = unlimited)
     */
    public static String readTextCapped(Path path, int maxChars) throws IOException {
        if (maxChars <= 0) {
            return Files.readString(path);
        }

        StringBuilder sb = new StringBuilder();
        try (var reader = Files.newBufferedReader(path)) {
            char[] buffer = new char[8192];
            int read;
            while ((read = reader.read(buffer)) != -1 && sb.length() < maxChars) {
                int remaining = maxChars - sb.length();
                int toRead = Math.min(read, remaining);
                sb.append(buffer, 0, toRead);
            }
        }
        return sb.toString();
    }

    public TextDataset(String path, int seqLen, Tokenizer tokenizer, int maxChars)
            throws IOException {
        this.seqLen = seqLen;
        this.rng = new Random(42);

        String raw = readTextCapped(Path.of(path), maxChars);

        this.tokens = tokenizer.encode(raw);

        // build index of valid start positions
        this.starts = new ArrayList<>();
        for (int i = 0; i + seqLen < tokens.length; i++) {
            starts.add(i);
        }

        System.out.println("[TextDataset] corpus=" + tokens.length
                + " tokens | windows=" + starts.size()
                + " | seqLen=" + seqLen);
    }

    /** Number of available sequence windows. */
    public int size() {
        return starts.size();
    }

    /** Total number of tokens in the corpus. */
    public int totalTokens() {
        return tokens.length;
    }

    /**
     * Sample a random batch of (input, target) pairs.
     * Target is input shifted by 1 position (next-token prediction).
     *
     * @return Batch with inputs[batchSize][seqLen] and targets[batchSize][seqLen]
     */
    public Batch getBatch(int batchSize) {
        int[][] inputs  = new int[batchSize][seqLen];
        int[][] targets = new int[batchSize][seqLen];

        for (int b = 0; b < batchSize; b++) {
            int start = starts.get(rng.nextInt(starts.size()));
            for (int t = 0; t < seqLen; t++) {
                inputs[b][t]  = tokens[start + t];
                targets[b][t] = tokens[start + t + 1];
            }
        }
        return new Batch(inputs, targets);
    }

    /**
     * Shuffle and iterate all sequences sequentially (for evaluation).
     * Returns up to maxBatches batches.
     */
    public List<Batch> iterateBatches(int batchSize, int maxBatches) {
        List<Integer> idx = new ArrayList<>(starts);
        Collections.shuffle(idx, rng);

        List<Batch> batches = new ArrayList<>();
        int bi = 0;
        while (bi + batchSize <= idx.size() && batches.size() < maxBatches) {
            int[][] inputs  = new int[batchSize][seqLen];
            int[][] targets = new int[batchSize][seqLen];
            for (int b = 0; b < batchSize; b++) {
                int start = idx.get(bi + b);
                for (int t = 0; t < seqLen; t++) {
                    inputs[b][t]  = tokens[start + t];
                    targets[b][t] = tokens[start + t + 1];
                }
            }
            batches.add(new Batch(inputs, targets));
            bi += batchSize;
        }
        return batches;
    }

    // -------------------------------------------------------------------------

    /** A single training batch: input token IDs and target token IDs. */
    public static class Batch {
        public final int[][] inputs;   // [batch, seqLen]
        public final int[][] targets;  // [batch, seqLen]

        public Batch(int[][] inputs, int[][] targets) {
            this.inputs = inputs;
            this.targets = targets;
        }
    }
}
