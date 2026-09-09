package fastdl.model;

import fastdl.layer.Layer;
import fastdl.tensor.Tensor;

import java.util.Collections;
import java.util.List;

/**
 * Learned Positional Encoding that injects position information into token embeddings.
 *
 * <p>Transformer self-attention is inherently position-agnostic — the attention mechanism
 * computes the same output regardless of the order of tokens. Positional encoding fixes
 * this by adding a position-dependent signal to each token's embedding before the first
 * Transformer block processes it.
 *
 * <h2>Learned vs. Sinusoidal</h2>
 * This implementation uses a <strong>learned</strong> positional embedding table
 * (shape {@code [seqLen, dModel]}), identical in structure to the token embedding.
 * Each position 0..seqLen-1 has its own trainable vector that is added element-wise
 * to the corresponding token embedding.
 *
 * <p>The original "Attention Is All You Need" paper used fixed sinusoidal encodings, but
 * GPT-2 and most modern LLMs use learned positional embeddings, which the model can
 * adapt to the specific positional patterns in the training data.
 *
 * <h2>Weight initialization</h2>
 * Position vectors are initialized from {@code N(0, 0.02²)}, the same as token
 * embeddings, following GPT-2 conventions.
 *
 * <h2>Shape contract</h2>
 * <ul>
 *   <li>Input:  {@code Tensor [batch, T, dModel]} — token embeddings from {@link Embedding}</li>
 *   <li>Output: {@code Tensor [batch, T, dModel]} — token + position embeddings</li>
 *   <li>T must be ≤ {@code seqLen} (the table's maximum length)</li>
 * </ul>
 *
 * <h2>Backward pass</h2>
 * Gradients accumulate into the position vectors for positions 0..T-1 that were
 * actually used in the forward pass. The input gradient is passed through unchanged
 * (positional encoding is a pure addition).
 *
 * <h2>Position in the pipeline</h2>
 * <pre>{@code
 * Tensor tok = tokenEmbed.forward(tokenIds);    // [B, T, dModel]
 * Tensor x   = posEncode.forward(tok);          // [B, T, dModel]  ← here
 * // → TransformerBlock(s)
 * }</pre>
 *
 * @see fastdl.model.Embedding
 * @see fastdl.model.GPTModel
 * @see fastdl.model.TransformerBlock
 */
public class PositionalEncoding implements Layer {

    private final int seqLen;
    private final int dModel;
    private final Tensor weight;   // [seqLen, dModel]

    private int lastBatch;
    private int lastT;

    public PositionalEncoding(int seqLen, int dModel) {
        this.seqLen = seqLen;
        this.dModel = dModel;
        // small random init
        this.weight = Tensor.randn(seqLen, dModel);
        float[] wd = weight.data();
        for (int i = 0; i < wd.length; i++) wd[i] *= 0.02f;
    }

    /**
     * Add positional embeddings to token embeddings.
     * @param input Tensor [batch, T, dModel]
     * @return Tensor [batch, T, dModel]
     */
    @Override
    public Tensor forward(Tensor input) {
        int[] shape = input.shape();
        int B = shape[0];
        int T = shape[1];
        this.lastBatch = B;
        this.lastT = T;

        Tensor out = new Tensor(B, T, dModel);
        float[] id = input.data();
        float[] wd = weight.data();
        float[] od = out.data();

        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                int inBase  = (b * T + t) * dModel;
                int posBase = t * dModel;
                int outBase = (b * T + t) * dModel;
                for (int d = 0; d < dModel; d++) {
                    od[outBase + d] = id[inBase + d] + wd[posBase + d];
                }
            }
        }
        return out;
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        float[] go = gradOutput.data();
        float[] wg = weight.grad();
        int B = lastBatch;
        int T = lastT;

        // accumulate grad into positional weight
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                int goBase  = (b * T + t) * dModel;
                int posBase = t * dModel;
                for (int d = 0; d < dModel; d++) {
                    wg[posBase + d] += go[goBase + d];
                }
            }
        }
        // grad w.r.t. input is unchanged (identity + pos)
        return gradOutput;
    }

    @Override
    public List<Tensor> parameters() {
        return List.of(weight);
    }

    public Tensor weight() { return weight; }
}
