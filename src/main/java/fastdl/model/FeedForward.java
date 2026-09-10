package fastdl.model;

import fastdl.layer.Dense;
import fastdl.layer.GELU;
import fastdl.layer.Layer;
import fastdl.ops.TensorOps;
import fastdl.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Position-wise Feed-Forward Network (FFN) — the second sub-layer inside every Transformer block.
 *
 * <p>After the self-attention sub-layer has allowed tokens to exchange information across
 * positions, the feed-forward block applies an identical, independent transformation to
 * each token's representation. Because it is applied identically and independently at
 * every position it is sometimes called a "position-wise" FFN.
 *
 * <h2>Architecture</h2>
 * <pre>
 *   h    = GELU( x · W₁ + b₁ )     W₁: [dModel, 4*dModel]
 *   out  = h · W₂ + b₂              W₂: [4*dModel, dModel]
 * </pre>
 * The hidden dimension is expanded to {@code 4 × dModel} (the "expansion factor").
 * This ratio of 4 comes from the original "Attention Is All You Need" paper and has
 * been adopted by GPT-2, GPT-3, and virtually all subsequent LLMs.
 *
 * <h2>Why GELU instead of ReLU?</h2>
 * GELU provides a smooth gradient for all input values, whereas ReLU has zero gradient
 * for negative inputs (dead neurons). GELU consistently outperforms ReLU on
 * language modelling benchmarks. See {@link fastdl.layer.GELU} for the full derivation.
 *
 * <h2>Parameter count</h2>
 * {@code 2 × (4 × dModel² + 4 × dModel + dModel² + dModel)}
 * ≈ {@code 8 × dModel²} — the FFN accounts for roughly ⅔ of a Transformer block's
 * total parameters when dModel is large.
 *
 * <h2>Shape contract</h2>
 * <ul>
 *   <li>Input:  {@code Tensor [batch, seqLen, dModel]}</li>
 *   <li>Output: {@code Tensor [batch, seqLen, dModel]}</li>
 * </ul>
 * Internally the tensor is flattened to {@code [batch*seqLen, dModel]} for the Dense
 * layers, then reshaped back. This is transparent to the caller.
 *
 * <h2>Usage inside TransformerBlock</h2>
 * <pre>{@code
 * FeedForward ff = new FeedForward(dModel=128);
 * // hidden dim = 512 internally
 * Tensor out  = ff.forward(normedInput);    // [B, T, 128]
 * Tensor grad = ff.backward(gradOut);       // [B, T, 128]
 * }</pre>
 *
 * @see fastdl.model.TransformerBlock
 * @see fastdl.layer.GELU
 * @see fastdl.layer.Dense
 */
public class FeedForward implements Layer {

    private final int dModel;
    private final Dense fc1;
    private final GELU gelu;
    private final Dense fc2;

    private int lastB, lastT;

    public FeedForward(int dModel) {
        this.dModel = dModel;
        int hidden = 4 * dModel;
        this.fc1  = new Dense(dModel, hidden);
        this.gelu = new GELU();
        this.fc2  = new Dense(hidden, dModel);
    }

    @Override
    public Tensor forward(Tensor input) {
        int[] shape = input.shape();
        lastB = shape[0];
        lastT = shape[1];

        Tensor flat = TensorOps.reshape(input, lastB * lastT, dModel);
        Tensor h    = fc1.forward(flat);
        Tensor hAct = gelu.forward(h);
        Tensor out  = fc2.forward(hAct);
        return TensorOps.reshape(out, lastB, lastT, dModel);
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        Tensor gradFlat = TensorOps.reshape(gradOutput, lastB * lastT, dModel);
        Tensor g2       = fc2.backward(gradFlat);
        Tensor gAct     = gelu.backward(g2);
        Tensor g1       = fc1.backward(gAct);
        return TensorOps.reshape(g1, lastB, lastT, dModel);
    }

    @Override
    public List<Tensor> parameters() {
        List<Tensor> p = new ArrayList<>();
        p.addAll(fc1.parameters());
        p.addAll(fc2.parameters());
        return p;
    }
}
