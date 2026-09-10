package fastdl.model;

import fastdl.layer.Dense;
import fastdl.layer.Layer;
import fastdl.layer.LayerNorm;
import fastdl.ops.TensorOps;
import fastdl.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * GPT-style causal language model — the top-level model class.
 *
 * <p>Implements the complete autoregressive Transformer architecture used in GPT-2
 * and similar models. Given a sequence of token IDs, the model predicts a probability
 * distribution over the entire vocabulary for the next token at every position.
 *
 * <h2>Full architecture (forward pass)</h2>
 * <pre>
 *   int[][] tokenIds  [B, T]
 *         │
 *         ├─ TokenEmbedding     [B, T] → [B, T, dModel]
 *         ├─ PositionalEncoding [B, T, dModel] → [B, T, dModel]  (adds position)
 *         │
 *         ├─ TransformerBlock × numLayers
 *         │     ├─ Pre-LN + CausalMultiHeadAttention + residual
 *         │     └─ Pre-LN + FeedForward + residual
 *         │
 *         ├─ Final LayerNorm   [B, T, dModel]
 *         └─ LM Head (Dense)  [B*T, dModel] → [B*T, vocabSize]   ← logits
 * </pre>
 *
 * <h2>Output format</h2>
 * The forward pass returns a flat {@code Tensor[B*T, vocabSize]} of raw unnormalized
 * logits. This flat layout matches the input expected by {@link fastdl.loss.CrossEntropyLoss}
 * without requiring an extra reshape in the training loop.
 *
 * <h2>Backward pass</h2>
 * Call {@link #backwardPass(fastdl.tensor.Tensor)} with the gradient tensor returned by
 * {@link fastdl.loss.CrossEntropyLoss#backward()}. The method propagates gradients
 * backwards through all layers in reverse order and accumulates them into each
 * parameter's {@code .grad()} array, ready for the optimizer.
 *
 * <h2>Parameter counting</h2>
 * {@link #paramCount()} sums the sizes of all trainable tensors. For the default
 * small config (dModel=128, 4 heads, 4 layers, seqLen=128, vocab≈102):
 * <pre>
 *   TokenEmbed:   102 × 128          =   13,056
 *   PosEncode:    128 × 128          =   16,384
 *   4 × Block:    4 × (4×128² + 8×128² + 2×128) ≈ 786,432
 *   Final LN:     2 × 128            =      256
 *   LM Head:      128 × 102 + 102    =   13,158
 *   ─────────────────────────────────────────
 *   Total:        ≈ 829,286 parameters
 * </pre>
 *
 * <h2>Usage (minimal training loop)</h2>
 * <pre>{@code
 * GPTModel model    = new GPTModel(GPTConfig.small(tokenizer.vocabSize()));
 * AdamW    optimizer = new AdamW(model.parameters(), 3e-4f);
 * CrossEntropyLoss loss = new CrossEntropyLoss();
 *
 * Tensor logits = model.forward(batch.inputs);
 * float  l      = loss.forward(logits, flatTargets);
 * optimizer.zeroGrad();
 * model.backwardPass(loss.backward());
 * optimizer.step();
 * }</pre>
 *
 * @see fastdl.model.GPTConfig
 * @see fastdl.training.Trainer
 * @see fastdl.generation.Generator
 * @see fastdl.loss.CrossEntropyLoss
 */
public class GPTModel implements Layer {

    private final GPTConfig config;

    private final Embedding tokenEmbed;
    private final PositionalEncoding posEncode;
    private final List<TransformerBlock> blocks;
    private final LayerNorm finalNorm;
    private final Dense lmHead;   // language model head

    private int lastB, lastT;
    private Tensor embedOut;     // after embedding + pos
    private Tensor blockOut;     // after all transformer blocks
    private Tensor normOut;      // after final LayerNorm

    public GPTModel(GPTConfig config) {
        this.config     = config;
        this.tokenEmbed = new Embedding(config.vocabSize, config.dModel);
        this.posEncode  = new PositionalEncoding(config.seqLen, config.dModel);

        this.blocks = new ArrayList<>();
        for (int i = 0; i < config.numLayers; i++) {
            blocks.add(new TransformerBlock(config.dModel, config.numHeads));
        }

        this.finalNorm = new LayerNorm(config.dModel);
        this.lmHead    = new Dense(config.dModel, config.vocabSize);

        // tie weights: lm head uses same weights as token embedding (common practice)
        // (skipped here to keep things simple — adds complexity in backward)
    }

    /**
     * Forward pass.
     * @param tokenIds [batch, seqLen] — integer token IDs
     * @return logits [batch * seqLen, vocabSize] — unnormalized scores per token
     */
    public Tensor forward(int[][] tokenIds) {
        lastB = tokenIds.length;
        lastT = tokenIds[0].length;

        // Embeddings
        Tensor te = tokenEmbed.forward(tokenIds);      // [B, T, dModel]
        embedOut   = posEncode.forward(te);            // [B, T, dModel]

        // Transformer blocks
        Tensor x = embedOut;
        for (TransformerBlock block : blocks) {
            x = block.forward(x);
        }
        blockOut = x;

        // Final norm
        normOut = finalNorm.forward(blockOut);         // [B, T, dModel]

        // LM head: flatten to [B*T, dModel] then project to vocab
        Tensor flat = TensorOps.reshape(normOut, lastB * lastT, config.dModel);
        return lmHead.forward(flat);                   // [B*T, vocabSize]
    }

    /**
     * Backward pass.
     * @param gradLogits [batch * seqLen, vocabSize] gradient from CrossEntropyLoss
     */
    public void backwardPass(Tensor gradLogits) {
        // LM head backward
        Tensor gradNormFlat = lmHead.backward(gradLogits);           // [B*T, dModel]
        Tensor gradNorm     = TensorOps.reshape(gradNormFlat, lastB, lastT, config.dModel);

        // Final LayerNorm backward
        Tensor gradBlock = finalNorm.backward(gradNorm);             // [B, T, dModel]

        // Transformer blocks backward (reverse order)
        Tensor g = gradBlock;
        for (int i = blocks.size() - 1; i >= 0; i--) {
            g = blocks.get(i).backward(g);
        }

        // Positional encoding backward
        Tensor gradEmbed = posEncode.backward(g);

        // Token embedding backward
        tokenEmbed.backward(gradEmbed);
    }

    /** Layer.forward() adapter — use forward(int[][]) in the training loop. */
    @Override
    public Tensor forward(Tensor input) {
        throw new UnsupportedOperationException("Use forward(int[][])");
    }

    /** Layer.backward() adapter — delegates to backwardPass(). */
    @Override
    public Tensor backward(Tensor gradOutput) {
        backwardPass(gradOutput);
        return gradOutput;
    }

    @Override
    public List<Tensor> parameters() {
        List<Tensor> p = new ArrayList<>();
        p.addAll(tokenEmbed.parameters());
        p.addAll(posEncode.parameters());
        for (TransformerBlock block : blocks) p.addAll(block.parameters());
        p.addAll(finalNorm.parameters());
        p.addAll(lmHead.parameters());
        return p;
    }

    public GPTConfig config() { return config; }

    /** Count total trainable parameters. */
    public long paramCount() {
        long count = 0;
        for (Tensor t : parameters()) count += t.size();
        return count;
    }

    @Override
    public String toString() {
        return "GPTModel(" + config + ", params=" + paramCount() + ")";
    }
}
