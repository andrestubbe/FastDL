package fastdl.model;

import fastdl.layer.Layer;
import fastdl.layer.LayerNorm;
import fastdl.ops.TensorOps;
import fastdl.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * A single Transformer Block implementing the Pre-LayerNorm (Pre-LN) architecture.
 *
 * <p>Each Transformer block consists of two sub-layers, each with its own residual
 * connection: a causal multi-head self-attention layer and a position-wise feed-forward
 * network. This is the fundamental repeating unit of GPT-style language models —
 * stacking N of these blocks creates a model of depth N.
 *
 * <h2>Pre-LN vs Post-LN</h2>
 * This implementation uses <strong>Pre-LayerNorm</strong> (normalize <em>before</em>
 * the sub-layer), as opposed to the original "Attention Is All You Need" design which
 * applies LayerNorm after the residual addition (Post-LN).
 *
 * <p>Pre-LN has been empirically shown to be more training-stable for small models and
 * shorter training runs, because the residual path always carries the unnormalized signal
 * directly to the next layer, preventing gradient vanishing in deep stacks.
 *
 * <h2>Computation graph (forward pass)</h2>
 * <pre>
 *   ┌── input ──────────────────────────────┐
 *   │                                       │ (residual)
 *   └→ LayerNorm₁ → Attention ─────────────┤ +
 *                                           ↓
 *   ┌── after_attn ─────────────────────────┐
 *   │                                       │ (residual)
 *   └→ LayerNorm₂ → FeedForward ───────────┤ +
 *                                           ↓
 *                                        output
 * </pre>
 *
 * <h2>Backward pass</h2>
 * Gradients flow through both branches of each residual connection. The identity
 * shortcut ensures that gradients can flow unimpeded through arbitrarily many blocks,
 * which is the key insight that makes deep Transformers trainable.
 *
 * <h2>Shape contract</h2>
 * <ul>
 *   <li>Input:  {@code Tensor [batch, seqLen, dModel]}</li>
 *   <li>Output: {@code Tensor [batch, seqLen, dModel]}</li>
 * </ul>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * TransformerBlock block = new TransformerBlock(dModel=128, numHeads=4);
 * Tensor out  = block.forward(x);       // [B, T, 128]
 * Tensor grad = block.backward(dOut);   // [B, T, 128]
 * }</pre>
 *
 * @see fastdl.model.GPTModel
 * @see fastdl.model.MultiHeadAttention
 * @see fastdl.model.FeedForward
 * @see fastdl.layer.LayerNorm
 */
public class TransformerBlock implements Layer {

    private final LayerNorm norm1;
    private final MultiHeadAttention attention;
    private final LayerNorm norm2;
    private final FeedForward feedForward;

    // saved residuals for backward
    private Tensor residual1;   // input to block (before attn branch)
    private Tensor residual2;   // after attn residual (before ff branch)
    private Tensor norm1Out;    // LayerNorm1 output
    private Tensor norm2Out;    // LayerNorm2 output
    private Tensor attnOut;     // Attention output (before residual add)
    private Tensor ffOut;       // FeedForward output (before residual add)

    public TransformerBlock(int dModel, int numHeads) {
        this.norm1       = new LayerNorm(dModel);
        this.attention   = new MultiHeadAttention(dModel, numHeads);
        this.norm2       = new LayerNorm(dModel);
        this.feedForward = new FeedForward(dModel);
    }

    @Override
    public Tensor forward(Tensor input) {
        // Branch 1: Attention
        residual1 = input;
        norm1Out  = norm1.forward(input);
        attnOut   = attention.forward(norm1Out);
        Tensor afterAttn = addResidual(attnOut, residual1);

        // Branch 2: FeedForward
        residual2 = afterAttn;
        norm2Out  = norm2.forward(afterAttn);
        ffOut     = feedForward.forward(norm2Out);
        return addResidual(ffOut, residual2);
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        // Branch 2 backward (FeedForward)
        Tensor gradFF   = feedForward.backward(gradOutput);
        Tensor gradNorm2 = norm2.backward(gradFF);
        // residual: grad passes through both branches
        Tensor gradAfterAttn = addTensors(gradNorm2, gradOutput);

        // Branch 1 backward (Attention)
        Tensor gradAttn  = attention.backward(gradAfterAttn);
        Tensor gradNorm1 = norm1.backward(gradAttn);
        return addTensors(gradNorm1, gradAfterAttn);
    }

    @Override
    public List<Tensor> parameters() {
        List<Tensor> p = new ArrayList<>();
        p.addAll(norm1.parameters());
        p.addAll(attention.parameters());
        p.addAll(norm2.parameters());
        p.addAll(feedForward.parameters());
        return p;
    }

    // -------------------------------------------------------------------------

    private static Tensor addResidual(Tensor a, Tensor b) {
        return addTensors(a, b);
    }

    private static Tensor addTensors(Tensor a, Tensor b) {
        float[] ad = a.data(), bd = b.data();
        Tensor out = new Tensor(a.shape());
        float[] od = out.data();
        for (int i = 0; i < ad.length; i++) od[i] = ad[i] + bd[i];
        return out;
    }
}
