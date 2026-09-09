package fastdl.generation;

import fastdl.model.GPTModel;
import fastdl.ops.TensorOps;
import fastdl.tensor.Tensor;
import fastdl.tokenizer.Tokenizer;

import java.util.Random;

/**
 * Autoregressive text generator for a trained {@link fastdl.model.GPTModel}.
 *
 * <p>Generation is performed by repeatedly feeding the current context into the model,
 * sampling the next token, and appending it to the sequence until the desired token
 * budget is reached or an EOS token is emitted. The generator supports both deterministic
 * greedy decoding and stochastic sampling with temperature and top-k filtering.
 *
 * <h2>Generation loop</h2>
 * <pre>{@code
 * Generator gen = new Generator(model, tokenizer);
 * String text = gen.generate("Once upon a time", 64, 0.9f, 20);
 * }</pre>
 *
 * <h2>Sampling modes</h2>
 * <ul>
 *   <li>greedy — highest-probability next token</li>
 *   <li>temperature — softens or sharpens the distribution</li>
 *   <li>top-k — keeps only the strongest candidates</li>
 * </ul>
 *
 * @see fastdl.model.GPTModel
 * @see fastdl.tokenizer.Tokenizer
 */
public class Generator {

    private final GPTModel model;
    private final Tokenizer tokenizer;
    private final int seqLen;
    private final Random rng;

    public Generator(GPTModel model, Tokenizer tokenizer) {
        this.model     = model;
        this.tokenizer = tokenizer;
        this.seqLen    = model.config().seqLen;
        this.rng       = new Random(System.nanoTime());
    }

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    /** Greedy generation — always pick the highest-probability next token. */
    public String generate(String prompt, int maxNewTokens) {
        return generate(prompt, maxNewTokens, 1.0f, 1);
    }

    /** Temperature sampling — temperature < 1 = sharper, > 1 = more random. */
    public String generate(String prompt, int maxNewTokens, float temperature) {
        return generate(prompt, maxNewTokens, temperature, 0);
    }

    /**
     * Full generation with temperature + top-K sampling.
     *
     * @param prompt       seed text
     * @param maxNewTokens how many new tokens to generate
     * @param temperature  softmax temperature (1.0 = neutral)
     * @param topK         top-K sampling (0 = disabled, use temperature only)
     * @return generated string including the prompt
     */
    public String generate(String prompt, int maxNewTokens,
                           float temperature, int topK) {
        int[] ctx = tokenizer.encode(prompt);
        int[] tokens = ctx.clone();

        for (int i = 0; i < maxNewTokens; i++) {
            // crop context to seqLen
            int start = Math.max(0, tokens.length - seqLen);
            int len   = tokens.length - start;
            int[] window = new int[len];
            System.arraycopy(tokens, start, window, 0, len);

            // forward — single batch of 1
            Tensor logits = model.forward(new int[][]{window});  // [len, vocab]
            // take logits of last token
            int vocabSize = model.config().vocabSize;
            float[] lastLogits = new float[vocabSize];
            System.arraycopy(logits.data(), (len - 1) * vocabSize, lastLogits, 0, vocabSize);

            // sample next token
            int nextToken = sample(lastLogits, temperature, topK);

            // append
            int[] newTokens = new int[tokens.length + 1];
            System.arraycopy(tokens, 0, newTokens, 0, tokens.length);
            newTokens[tokens.length] = nextToken;
            tokens = newTokens;

            // stop at EOS
            if (nextToken == tokenizer.eosTokenId()) break;
        }

        // decode only new tokens
        int[] generated = new int[tokens.length - ctx.length];
        System.arraycopy(tokens, ctx.length, generated, 0, generated.length);
        return prompt + tokenizer.decode(generated);
    }

    // -------------------------------------------------------------------------
    // Sampling strategies
    // -------------------------------------------------------------------------

    private int sample(float[] logits, float temperature, int topK) {
        if (temperature <= 0f || (topK == 1)) {
            return argmax(logits);
        }

        // apply temperature
        float[] scaled = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            scaled[i] = logits[i] / temperature;
        }

        // top-K masking
        if (topK > 1 && topK < scaled.length) {
            float kThreshold = kthLargest(scaled, topK);
            for (int i = 0; i < scaled.length; i++) {
                if (scaled[i] < kThreshold) scaled[i] = Float.NEGATIVE_INFINITY;
            }
        }

        // softmax
        float max = Float.NEGATIVE_INFINITY;
        for (float v : scaled) if (v > max) max = v;
        float sum = 0f;
        float[] probs = new float[scaled.length];
        for (int i = 0; i < scaled.length; i++) {
            probs[i] = (float) Math.exp(scaled[i] - max);
            sum += probs[i];
        }
        for (int i = 0; i < probs.length; i++) probs[i] /= sum;

        // multinomial sample
        float u = rng.nextFloat();
        float cumulative = 0f;
        for (int i = 0; i < probs.length; i++) {
            cumulative += probs[i];
            if (u < cumulative) return i;
        }
        return probs.length - 1;
    }

    private static int argmax(float[] arr) {
        int best = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[best]) best = i;
        }
        return best;
    }

    /** Returns the k-th largest value in arr (partially sorted). */
    private static float kthLargest(float[] arr, int k) {
        float[] copy = arr.clone();
        java.util.Arrays.sort(copy);
        return copy[copy.length - k];
    }
}
