package fastdl.training;

import fastdl.data.TextDataset;
import fastdl.loss.CrossEntropyLoss;
import fastdl.model.GPTModel;
import fastdl.optim.AdamW;
import fastdl.tensor.Tensor;

import java.io.*;
import java.nio.file.*;
import java.util.List;
import java.util.function.Consumer;

/**
 * Training loop for autoregressive GPT-style language models.
 *
 * <p>The {@code Trainer} orchestrates the full supervised optimization cycle for
 * next-token prediction. It takes a {@link fastdl.model.GPTModel}, a
 * {@link fastdl.data.TextDataset}, an {@link fastdl.optim.AdamW} optimizer, and a
 * {@link TrainerConfig}, then runs the training loop for {@code config.maxIter} steps.
 *
 * <h2>Per-step cycle</h2>
 * <ol>
 *   <li>Sample a random batch of {@code (inputs, targets)} from the dataset</li>
 *   <li>Forward pass: {@code logits = model.forward(inputs)} — shape {@code [B*T, vocab]}</li>
 *   <li>Loss: {@code loss = CrossEntropyLoss.forward(logits, flatTargets)}</li>
 *   <li>Zero gradients: {@code optimizer.zeroGrad()}</li>
 *   <li>Backward pass: {@code model.backwardPass(criterion.backward())}</li>
 *   <li>Gradient clipping: clip global norm to {@code config.gradClip}</li>
 *   <li>Optimizer step: {@code optimizer.step()}</li>
 * </ol>
 *
 * <h2>Evaluation</h2>
 * Every {@code config.evalInterval} steps the trainer calls {@link #evaluate()} which
 * averages the cross-entropy loss over {@code config.evalBatches} random batches.
 * The result is passed to the optional {@link #setOnStep(java.util.function.Consumer)}
 * callback as a {@link StepInfo} record for progress display.
 *
 * <h2>Gradient clipping</h2>
 * If {@code config.gradClip > 0}, the global L2 norm of all gradients is computed
 * and all gradients are scaled down proportionally if the norm exceeds the threshold.
 * This is the standard technique to prevent exploding gradients in deep Transformers.
 *
 * <h2>Checkpointing</h2>
 * If {@code config.checkpointDir} is set, the model's parameter arrays are serialized
 * to a binary {@code .dat} file after each evaluation. Checkpoints can be loaded back
 * via {@link #loadCheckpoint(String)}.
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * Trainer trainer = new Trainer(model, dataset, optimizer, TrainerConfig.demo());
 * trainer.setOnStep(info ->
 *     System.out.printf("[%d/%d] train=%.4f eval=%.4f%n",
 *         info.step, info.totalSteps, info.trainLoss, info.evalLoss));
 * trainer.train();
 * }</pre>
 *
 * @see TrainerConfig
 * @see fastdl.model.GPTModel
 * @see fastdl.data.TextDataset
 * @see fastdl.optim.AdamW
 */
public class Trainer {

    private final GPTModel model;
    private final TextDataset dataset;
    private final AdamW optimizer;
    private final TrainerConfig config;
    private final CrossEntropyLoss criterion;

    private Consumer<StepInfo> onStep;   // progress callback

    public Trainer(GPTModel model, TextDataset dataset,
                   AdamW optimizer, TrainerConfig config) {
        this.model     = model;
        this.dataset   = dataset;
        this.optimizer = optimizer;
        this.config    = config;
        this.criterion = new CrossEntropyLoss();
    }

    /** Register a callback invoked after each evalInterval step. */
    public void setOnStep(Consumer<StepInfo> callback) {
        this.onStep = callback;
    }

    // -------------------------------------------------------------------------

    public void train() {
        for (int step = 1; step <= config.maxIter; step++) {
            // --- forward ---
            TextDataset.Batch batch = dataset.getBatch(config.batchSize);
            int B = config.batchSize;
            int T = batch.inputs[0].length;

            Tensor logits = model.forward(batch.inputs);       // [B*T, vocab]

            // flatten targets to 1-D
            int[] targets = flattenTargets(batch.targets, B, T);
            float loss = criterion.forward(logits, targets);

            // --- backward ---
            optimizer.zeroGrad();
            Tensor gradLogits = criterion.backward();
            model.backwardPass(gradLogits);

            // gradient clipping
            if (config.gradClip > 0f) {
                clipGradients(model.parameters(), config.gradClip);
            }

            optimizer.step();

            // --- reporting ---
            if (step % config.evalInterval == 0 || step == 1) {
                float evalLoss = evaluate();
                if (onStep != null) {
                    onStep.accept(new StepInfo(step, config.maxIter, loss, evalLoss));
                }
                if (config.checkpointDir != null) {
                    saveCheckpoint(step);
                }
            }
        }
    }

    /** Compute average loss over a few held-out batches. */
    public float evaluate() {
        double sum = 0.0;
        int n = config.evalBatches;
        int T = model.config().seqLen;
        for (int i = 0; i < n; i++) {
            TextDataset.Batch batch = dataset.getBatch(config.batchSize);
            Tensor logits = model.forward(batch.inputs);
            int[] targets = flattenTargets(batch.targets, config.batchSize, T);
            sum += criterion.forward(logits, targets);
        }
        return (float) (sum / n);
    }

    // -------------------------------------------------------------------------

    private static int[] flattenTargets(int[][] targets, int B, int T) {
        int[] flat = new int[B * T];
        for (int b = 0; b < B; b++) {
            System.arraycopy(targets[b], 0, flat, b * T, T);
        }
        return flat;
    }

    private static void clipGradients(List<Tensor> params, float maxNorm) {
        double totalNorm = 0.0;
        for (Tensor p : params) {
            float[] g = p.grad();
            for (float v : g) totalNorm += (double) v * v;
        }
        totalNorm = Math.sqrt(totalNorm);
        if (totalNorm > maxNorm) {
            float scale = (float) (maxNorm / totalNorm);
            for (Tensor p : params) {
                float[] g = p.grad();
                for (int i = 0; i < g.length; i++) g[i] *= scale;
            }
        }
    }

    private void saveCheckpoint(int step) {
        try {
            Path dir = Path.of(config.checkpointDir);
            Files.createDirectories(dir);
            Path file = dir.resolve("checkpoint_" + step + ".dat");
            try (DataOutputStream dos = new DataOutputStream(
                    new BufferedOutputStream(Files.newOutputStream(file)))) {
                for (Tensor p : model.parameters()) {
                    float[] data = p.data();
                    dos.writeInt(data.length);
                    for (float v : data) dos.writeFloat(v);
                }
            }
        } catch (IOException e) {
            System.err.println("[Trainer] Checkpoint save failed: " + e.getMessage());
        }
    }

    public void loadCheckpoint(String path) {
        try (DataInputStream dis = new DataInputStream(
                new BufferedInputStream(Files.newInputStream(Path.of(path))))) {
            for (Tensor p : model.parameters()) {
                int len = dis.readInt();
                float[] data = p.data();
                for (int i = 0; i < len; i++) data[i] = dis.readFloat();
            }
            System.out.println("[Trainer] Loaded checkpoint: " + path);
        } catch (IOException e) {
            System.err.println("[Trainer] Checkpoint load failed: " + e.getMessage());
        }
    }

    // -------------------------------------------------------------------------

    /** Data class for step progress callbacks. */
    public static class StepInfo {
        public final int step;
        public final int totalSteps;
        public final float trainLoss;
        public final float evalLoss;

        public StepInfo(int step, int totalSteps, float trainLoss, float evalLoss) {
            this.step       = step;
            this.totalSteps = totalSteps;
            this.trainLoss  = trainLoss;
            this.evalLoss   = evalLoss;
        }

        public float progressPct() {
            return 100f * step / totalSteps;
        }
    }
}
