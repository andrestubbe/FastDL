package fastdl.loss;

import fastdl.tensor.Tensor;

/**
 * Numerically stable Cross-Entropy Loss for next-token prediction in language models.
 *
 * <p>This is the standard loss function used to train GPT-style autoregressive language
 * models. For each position in the sequence the model outputs a distribution over the
 * entire vocabulary (logits), and the loss measures how surprised the model is at the
 * correct next token.
 *
 * <h2>Mathematical definition</h2>
 * <pre>
 *   softmax_i  = exp(logit_i - max) / Σ exp(logit_j - max)   (numerically stable)
 *   loss_t     = -log(softmax[target_t])                       (negative log-likelihood)
 *   L          = mean(loss_t)  over all t in batch*seqLen
 * </pre>
 *
 * <h2>Gradient (backward pass)</h2>
 * The gradient with respect to the logits has a beautifully simple closed form:
 * <pre>
 *   dL/dlogit_i = (softmax_i - 1{i == target}) / N
 * </pre>
 * where N = batch * seqLen. This is why Cross-Entropy + Softmax is so popular:
 * the combined gradient is just softmax minus the one-hot target.
 *
 * <h2>Numerical stability</h2>
 * The log-sum-exp trick is applied: before computing exp(), the row maximum is
 * subtracted. This prevents overflow for large logit values without changing the
 * mathematical result.
 *
 * <h2>Input / output contract</h2>
 * <ul>
 *   <li><b>logits</b>  — {@code Tensor[N, vocabSize]} — raw unnormalized scores (NOT softmaxed)</li>
 *   <li><b>targets</b> — {@code int[N]} — correct token index per row, range [0, vocabSize)</li>
 *   <li><b>returns</b> — scalar float, average NLL loss over all N rows</li>
 * </ul>
 * N = batchSize * seqLen (the logits are pre-flattened by GPTModel).
 *
 * <h2>Usage in the training loop</h2>
 * <pre>{@code
 * CrossEntropyLoss criterion = new CrossEntropyLoss();
 *
 * Tensor logits = model.forward(tokenIds);       // [B*T, vocabSize]
 * float  loss   = criterion.forward(logits, targets);
 *
 * optimizer.zeroGrad();
 * model.backwardPass(criterion.backward());      // [B*T, vocabSize]  grad
 * optimizer.step();
 * }</pre>
 *
 * @see fastdl.training.Trainer
 * @see fastdl.model.GPTModel
 */
public class CrossEntropyLoss {

    private float[] lastSoftmax;
    private int[] lastTargets;
    private int rows;
    private int vocabSize;

    /**
     * Forward pass.
     * @param logits  2-D Tensor [N, V]  where N = batch*seq, V = vocabSize
     * @param targets int array of length N with the correct token index per row
     * @return scalar average cross-entropy loss
     */
    public float forward(Tensor logits, int[] targets) {
        int[] shape = logits.shape();
        this.rows = shape[0];
        this.vocabSize = shape[1];
        this.lastTargets = targets;
        this.lastSoftmax = new float[rows * vocabSize];

        float[] ld = logits.data();
        double totalLoss = 0.0;

        for (int r = 0; r < rows; r++) {
            int base = r * vocabSize;

            // log-sum-exp for numerical stability
            float max = Float.NEGATIVE_INFINITY;
            for (int v = 0; v < vocabSize; v++) {
                if (ld[base + v] > max) max = ld[base + v];
            }

            float sumExp = 0f;
            for (int v = 0; v < vocabSize; v++) {
                float e = (float) Math.exp(ld[base + v] - max);
                lastSoftmax[base + v] = e;
                sumExp += e;
            }
            for (int v = 0; v < vocabSize; v++) {
                lastSoftmax[base + v] /= sumExp;
            }

            int t = targets[r];
            totalLoss += -Math.log(Math.max(lastSoftmax[base + t], 1e-12f));
        }

        return (float) (totalLoss / rows);
    }

    /**
     * Backward pass — returns gradient w.r.t. logits [N, V].
     * dL/dlogits = (softmax - one_hot_target) / N
     */
    public Tensor backward() {
        Tensor grad = new Tensor(rows, vocabSize);
        float[] gd = grad.data();
        float scale = 1f / rows;

        System.arraycopy(lastSoftmax, 0, gd, 0, rows * vocabSize);
        for (int r = 0; r < rows; r++) {
            gd[r * vocabSize + lastTargets[r]] -= 1f;
        }
        for (int i = 0; i < gd.length; i++) {
            gd[i] *= scale;
        }
        return grad;
    }
}
