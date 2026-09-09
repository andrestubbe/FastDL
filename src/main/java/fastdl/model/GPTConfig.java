package fastdl.model;

/**
 * Immutable configuration record for a GPT-style causal language model.
 *
 * <p>All architectural hyperparameters of a {@link GPTModel} are bundled in this
 * record and validated at construction time. Pass one {@code GPTConfig} instance
 * to {@link GPTModel#GPTModel(GPTConfig)} to create a fully wired model.
 *
 * <h2>Parameters</h2>
 * <ul>
 *   <li><b>vocabSize</b>  — total number of distinct tokens. Must equal
 *       {@link fastdl.tokenizer.Tokenizer#vocabSize()} of the tokenizer used.</li>
 *   <li><b>seqLen</b>     — context window length in tokens. The model can attend to
 *       at most {@code seqLen} previous tokens. Longer seqLen = more context but
 *       quadratically more memory in the attention layers.</li>
 *   <li><b>dModel</b>     — embedding / hidden dimension. All internal representations
 *       have this width. Typical values: 64 (tiny), 128 (small), 256–512 (medium).</li>
 *   <li><b>numHeads</b>   — number of attention heads in each
 *       {@link fastdl.model.MultiHeadAttention} layer. Must divide {@code dModel}
 *       evenly. Each head operates on {@code dModel / numHeads} dimensions.</li>
 *   <li><b>numLayers</b>  — number of stacked {@link fastdl.model.TransformerBlock}s.
 *       More layers = more representational capacity but slower training.</li>
 *   <li><b>dropout</b>    — dropout rate [0, 1). Currently unused in the pure-JVM
 *       implementation (no random mask applied); reserved for future GPU path.</li>
 * </ul>
 *
 * <h2>Preset configurations</h2>
 * <ul>
 *   <li>{@link #small(int)} — 128-dim, 4 heads, 4 layers, seqLen=128 — ~1–3 M params</li>
 *   <li>{@link #medium(int)} — 256-dim, 4 heads, 6 layers, seqLen=256 — ~5–15 M params</li>
 * </ul>
 *
 * <h2>Validation</h2>
 * The constructor throws {@link IllegalArgumentException} if {@code dModel % numHeads != 0},
 * because each attention head requires an equal slice of the embedding dimension.
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * GPTConfig config = GPTConfig.small(tokenizer.vocabSize());
 * GPTModel  model  = new GPTModel(config);
 * System.out.println(model.paramCount() + " parameters");
 * }</pre>
 *
 * @see GPTModel
 * @see fastdl.model.MultiHeadAttention
 * @see fastdl.model.TransformerBlock
 */
public final class GPTConfig {

    public final int vocabSize;   // number of tokens
    public final int seqLen;      // context window / max sequence length
    public final int dModel;      // embedding dimension
    public final int numHeads;    // number of attention heads (dModel must be divisible)
    public final int numLayers;   // number of transformer blocks
    public final float dropout;   // dropout rate (0 = disabled in this pure-JVM impl)

    public GPTConfig(int vocabSize, int seqLen, int dModel,
                     int numHeads, int numLayers, float dropout) {
        if (dModel % numHeads != 0) {
            throw new IllegalArgumentException(
                "dModel (" + dModel + ") must be divisible by numHeads (" + numHeads + ")");
        }
        this.vocabSize = vocabSize;
        this.seqLen    = seqLen;
        this.dModel    = dModel;
        this.numHeads  = numHeads;
        this.numLayers = numLayers;
        this.dropout   = dropout;
    }

    /** Compact demo config: 128-dim, 4 heads, 4 layers, seqLen=128 */
    public static GPTConfig small(int vocabSize) {
        return new GPTConfig(vocabSize, 128, 128, 4, 4, 0.0f);
    }

    /** Slightly larger config: 256-dim, 4 heads, 6 layers, seqLen=256 */
    public static GPTConfig medium(int vocabSize) {
        return new GPTConfig(vocabSize, 256, 256, 4, 6, 0.1f);
    }

    @Override
    public String toString() {
        return "GPTConfig{vocab=" + vocabSize + ", seq=" + seqLen
             + ", d=" + dModel + ", heads=" + numHeads
             + ", layers=" + numLayers + "}";
    }
}
