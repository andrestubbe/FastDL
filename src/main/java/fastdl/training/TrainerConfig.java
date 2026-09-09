package fastdl.training;

/**
 * Training configuration for a FastDL language-model run.
 *
 * <p>TrainerConfig bundles the most important hyperparameters used by the training loop
 * in {@link fastdl.training.Trainer}: batch size, learning rate, optimizer settings,
 * evaluation interval, and optional checkpoint output directory.
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
