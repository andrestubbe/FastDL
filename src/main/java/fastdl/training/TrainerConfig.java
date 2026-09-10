package fastdl.training;

/**
 * Immutable configuration bundle for the {@link Trainer} training loop.
 *
 * <p>All hyperparameters that control the duration, frequency, and behaviour of
 * a training run are gathered here. Pass a {@code TrainerConfig} instance to
 * {@link Trainer} at construction time — it is not changed during training.
 *
 * <h2>Parameters</h2>
 * <ul>
 *   <li><b>batchSize</b>     — number of sequences per gradient update step.
 *       Larger batches = more stable gradients but more RAM and slower steps.</li>
 *   <li><b>maxIter</b>       — total number of optimizer steps (not epochs).
 *       Each step consumes {@code batchSize} random windows from the corpus.</li>
 *   <li><b>evalInterval</b>  — evaluate (compute eval loss) every N steps.
 *       Also controls the frequency of progress callbacks and checkpoint saves.</li>
 *   <li><b>evalBatches</b>   — number of random batches averaged for the eval loss.
 *       Higher = smoother eval estimate but slower evaluation pause.</li>
 *   <li><b>learningRate</b>  — AdamW step size. Typical range: 1e-4 to 3e-3.
 *       Too high → divergence. Too low → slow convergence.</li>
 *   <li><b>weightDecay</b>   — L2 regularisation coefficient (decoupled, AdamW style).
 *       0.01 is the GPT-2 default.</li>
 *   <li><b>gradClip</b>      — global gradient norm clipping threshold.
 *       Prevents exploding gradients. 1.0 is the standard value. Set to 0 to disable.</li>
 *   <li><b>checkpointDir</b> — directory for saving parameter snapshots every
 *       {@code evalInterval} steps. {@code null} disables checkpointing.</li>
 * </ul>
 *
 * <h2>Preset configurations</h2>
 * <ul>
 *   <li>{@link #demo()} — small, fast configuration intended for quick experiments</li>
 *   <li>{@link #standard()} — slightly larger config for realistic runs</li>
 * </ul>
 *
 * @see fastdl.training.Trainer
 * @see fastdl.optim.AdamW
 */
public final class TrainerConfig {

    public final int batchSize;
    public final int maxIter;           // total training steps
    public final int evalInterval;      // steps between loss prints
    public final int evalBatches;       // batches to average for eval loss
    public final float learningRate;
    public final float weightDecay;
    public final float gradClip;        // max gradient norm (0 = disabled)
    public final String checkpointDir;  // null = no checkpointing

    public TrainerConfig(int batchSize, int maxIter, int evalInterval,
                         int evalBatches, float learningRate, float weightDecay,
                         float gradClip, String checkpointDir) {
        this.batchSize      = batchSize;
        this.maxIter        = maxIter;
        this.evalInterval   = evalInterval;
        this.evalBatches    = evalBatches;
        this.learningRate   = learningRate;
        this.weightDecay    = weightDecay;
        this.gradClip       = gradClip;
        this.checkpointDir  = checkpointDir;
    }

    /** Quick demo config — fast to run, visible learning. */
    public static TrainerConfig demo() {
        return new TrainerConfig(
            /*batchSize*/    2,
            /*maxIter*/      200,
            /*evalInterval*/ 20,
            /*evalBatches*/  4,
            /*lr*/           3e-4f,
            /*weightDecay*/  0.01f,
            /*gradClip*/     1.0f,
            /*ckptDir*/      null
        );
    }

    /** Standard small-model config. */
    public static TrainerConfig standard() {
        return new TrainerConfig(
            /*batchSize*/    4,
            /*maxIter*/      2000,
            /*evalInterval*/ 100,
            /*evalBatches*/  10,
            /*lr*/           3e-4f,
            /*weightDecay*/  0.01f,
            /*gradClip*/     1.0f,
            /*ckptDir*/      "checkpoints"
        );
    }
}
