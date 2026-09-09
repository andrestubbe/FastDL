package fastdl.model;

import fastdl.layer.Dense;
import fastdl.layer.Layer;
import fastdl.layer.LayerNorm;
import fastdl.ops.TensorOps;
import fastdl.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * GPT-style causal language model.
 *
 * Architecture:
 *   TokenEmbedding + PositionalEncoding
 *   -> N x TransformerBlock
 *   -> LayerNorm
 *   -> Linear head [dModel -> vocabSize]   (logits)
 *
 * forward(int[][] tokenIds) -> logits Tensor [batch * seqLen, vocabSize]
 * The flat output is used directly by CrossEntropyLoss.
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
