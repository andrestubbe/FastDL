package fastdl.model;

import fastdl.layer.Layer;
import fastdl.tensor.Tensor;

import java.util.List;

/**
 * Learnable token embedding table that maps integer token IDs to dense vectors.
 *
 * <p>The embedding layer is the first trainable component in any language model.
 * It converts a sequence of discrete token IDs (integers) into continuous-valued
 * dense vectors of size {@code dModel}. These vectors capture semantic and syntactic
 * relationships between tokens — tokens that appear in similar contexts will have
 * similar embedding vectors after training.
 *
 * <h2>Internal structure</h2>
 * The embedding is a weight matrix {@code W} of shape {@code [vocabSize, dModel]}.
 * Looking up token ID {@code i} simply copies row {@code i} of {@code W} into the
 * output. This is equivalent to a one-hot vector times the weight matrix, but
 * implemented as a direct array copy for efficiency.
 *
 * <h2>Weight initialization</h2>
 * Weights are initialized from a standard normal distribution scaled by 0.02,
 * following the GPT-2 paper recommendation: {@code W ~ N(0, 0.02²)}.
 *
 * <h2>Backward pass</h2>
 * The gradient update accumulates into the rows of the weight matrix that were
 * accessed during the forward pass. Rows corresponding to tokens not in the current
 * batch receive zero gradient (and therefore no weight update). This is equivalent to
 * PyTorch's {@code nn.Embedding} with {@code sparse=False}.
 *
 * <h2>Shape contract</h2>
 * <ul>
 *   <li>Input:  {@code int[][] tokenIds} of shape {@code [batch, seqLen]}</li>
 *   <li>Output: {@code Tensor} of shape {@code [batch, seqLen, dModel]}</li>
 * </ul>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * Embedding embed = new Embedding(vocabSize=102, dModel=128);
 * int[][] tokens = {{5, 12, 33, 7}, {2, 44, 18, 9}};  // [batch=2, seqLen=4]
 * Tensor out = embed.forward(tokens);                  // [2, 4, 128]
 * }</pre>
 *
 * @see fastdl.model.GPTModel
 * @see fastdl.model.PositionalEncoding
 * @see fastdl.tokenizer.CharTokenizer
 */
public class Embedding implements Layer {

    private final int vocabSize;
    private final int dModel;
    private final Tensor weight;   // [vocabSize, dModel]

    private int[][] lastIds;       // saved for backward

    public Embedding(int vocabSize, int dModel) {
        this.vocabSize = vocabSize;
        this.dModel = dModel;
        // small-std init (similar to PyTorch default)
        this.weight = Tensor.randn(vocabSize, dModel);
        float[] wd = weight.data();
        for (int i = 0; i < wd.length; i++) wd[i] *= 0.02f;
    }

    /**
     * Forward for a 2-D batch of token IDs.
     * @param tokenIds [batch, seqLen]
     * @return Tensor [batch, seqLen, dModel]
     */
    public Tensor forward(int[][] tokenIds) {
        this.lastIds = tokenIds;
        int B = tokenIds.length;
        int T = tokenIds[0].length;
        Tensor out = new Tensor(B, T, dModel);
        float[] od = out.data();
        float[] wd = weight.data();
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                int id = tokenIds[b][t];
                int srcBase = id * dModel;
                int dstBase = (b * T + t) * dModel;
                System.arraycopy(wd, srcBase, od, dstBase, dModel);
            }
        }
        return out;
    }

    /** Convenience for single sequence. */
    public Tensor forward(int[] tokenIds) {
        return forward(new int[][]{tokenIds});
    }

    /** Layer.forward() adapter — not used in the transformer pipeline directly. */
    @Override
    public Tensor forward(Tensor input) {
        throw new UnsupportedOperationException("Use forward(int[][])");
    }

    /**
     * Backward: accumulate gradients into embedding weight table.
     * @param gradOutput [batch, seqLen, dModel]
     */
    @Override
    public Tensor backward(Tensor gradOutput) {
        float[] go = gradOutput.data();
        float[] wg = weight.grad();
        int B = lastIds.length;
        int T = lastIds[0].length;
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                int id = lastIds[b][t];
                int srcBase = (b * T + t) * dModel;
                int dstBase = id * dModel;
                for (int d = 0; d < dModel; d++) {
                    wg[dstBase + d] += go[srcBase + d];
                }
            }
        }
        return gradOutput; // grad w.r.t. input ids is not meaningful
    }

    @Override
    public List<Tensor> parameters() {
        return List.of(weight);
    }

    public Tensor weight() { return weight; }
    public int vocabSize() { return vocabSize; }
    public int dModel()    { return dModel; }
}
