package fastdl.model;

import fastdl.layer.Dense;
import fastdl.layer.Layer;
import fastdl.ops.TensorOps;
import fastdl.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Causal Multi-Head Self-Attention — the core computational unit of the Transformer.
 *
 * <p>Self-attention allows every token in a sequence to "look at" all previous tokens
 * and selectively aggregate their representations. The "multi-head" variant runs
 * {@code numHeads} independent attention functions in parallel, each operating on a
 * {@code dHead = dModel / numHeads} dimensional subspace, then concatenates their outputs.
 * This lets the model simultaneously attend to information from different representation
 * subspaces at different positions.
 *
 * <h2>Architecture (per forward pass)</h2>
 * <ol>
 *   <li>Project input [B, T, dModel] → Q, K, V each [B*T, dModel] via Dense layers</li>
 *   <li>For each head h (0..numHeads-1), slice the dHead-wide subspace</li>
 *   <li>Compute scaled dot-product attention scores: {@code score[q,k] = Q[q]·K[k] / √dHead}</li>
 *   <li>Apply <b>causal mask</b>: set score[q,k] = -∞ for k > q (no future token peeking)</li>
 *   <li>Softmax over keys: {@code attn[q,k] = softmax(score[q,:])[k]}</li>
 *   <li>Weighted sum of V: {@code out[q] = Σ_k attn[q,k] · V[k]}</li>
 *   <li>Project concatenated head outputs back to dModel via an output Dense layer</li>
 * </ol>
 *
 * <h2>Causal masking</h2>
 * The upper triangle of the attention score matrix is filled with {@code -∞} before
 * softmax. After softmax these entries become 0, so each query position can only
 * attend to itself and earlier positions. This is what makes the model autoregressive —
 * the prediction at position t is solely a function of tokens 0..t.
 *
 * <h2>Parameter count</h2>
 * {@code 4 × (dModel² + dModel)} — four Dense layers (Q, K, V, output projection)
 * each of shape [dModel, dModel] with a bias of size dModel.
 *
 * <h2>Shape contract</h2>
 * <ul>
 *   <li>Input:  {@code Tensor [batch, seqLen, dModel]}</li>
 *   <li>Output: {@code Tensor [batch, seqLen, dModel]}</li>
 * </ul>
 *
 * <h2>Usage inside TransformerBlock</h2>
 * <pre>{@code
 * MultiHeadAttention attn = new MultiHeadAttention(dModel=128, numHeads=4);
 * // dHead = 32 per head
 * Tensor out = attn.forward(normedInput);   // [B, T, 128]
 * Tensor grad = attn.backward(gradOut);     // [B, T, 128]
 * }</pre>
 *
 * @see fastdl.model.TransformerBlock
 * @see fastdl.model.GPTConfig
 * @see fastdl.layer.Dense
 */
public class MultiHeadAttention implements Layer {

    private final int dModel;
    private final int numHeads;
    private final int dHead;

    private final Dense qProj;   // [dModel -> dModel]
    private final Dense kProj;
    private final Dense vProj;
    private final Dense outProj; // [dModel -> dModel]

    // saved for backward
    private int lastB, lastT;
    private Tensor lastInput;
    private Tensor lastQ, lastK, lastV;   // pre-head-split projections
    private Tensor lastAttnWeights;       // [B*H, T, T]  softmax output
    private Tensor lastAttnOut;           // [B, T, dModel] before outProj

    public MultiHeadAttention(int dModel, int numHeads) {
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.dHead = dModel / numHeads;
        this.qProj   = new Dense(dModel, dModel);
        this.kProj   = new Dense(dModel, dModel);
        this.vProj   = new Dense(dModel, dModel);
        this.outProj = new Dense(dModel, dModel);
    }

    @Override
    public Tensor forward(Tensor input) {
        this.lastInput = input;
        int[] shape = input.shape();
        int B = shape[0];
        int T = shape[1];
        this.lastB = B;
        this.lastT = T;

        // Flatten to [B*T, dModel] for Dense layers
        Tensor flat = TensorOps.reshape(input, B * T, dModel);

        Tensor Q = qProj.forward(flat);   // [B*T, dModel]
        Tensor K = kProj.forward(flat);
        Tensor V = vProj.forward(flat);
        lastQ = Q; lastK = K; lastV = V;

        // Reshape + transpose to [B, numHeads, T, dHead]
        // then treat each head's [T, dHead] independently
        Tensor attnOut = scaledDotProductAttention(Q, K, V, B, T);
        lastAttnOut = attnOut;

        // Flatten and project output
        Tensor flatOut = TensorOps.reshape(attnOut, B * T, dModel);
        Tensor out = outProj.forward(flatOut);   // [B*T, dModel]
        return TensorOps.reshape(out, B, T, dModel);
    }

    /**
     * Scaled dot-product attention with causal mask.
     * Q, K, V: [B*T, dModel]
     * Returns: [B, T, dModel]
     */
    private Tensor scaledDotProductAttention(Tensor Q, Tensor K, Tensor V,
                                              int B, int T) {
        float scale = (float) (1.0 / Math.sqrt(dHead));
        float[] qd = Q.data();
        float[] kd = K.data();
        float[] vd = V.data();

        // Result accumulator [B, T, dModel]
        Tensor out = new Tensor(B, T, dModel);
        float[] od = out.data();

        // attn weights for backward [B, numHeads, T, T]
        lastAttnWeights = new Tensor(B, numHeads, T, T);
        float[] awd = lastAttnWeights.data();

        for (int b = 0; b < B; b++) {
            for (int h = 0; h < numHeads; h++) {
                int hOff = h * dHead;

                // Compute attention scores [T, T]
                float[] scores = new float[T * T];
                for (int qi = 0; qi < T; qi++) {
                    int qBase = (b * T + qi) * dModel + hOff;
                    for (int ki = 0; ki < T; ki++) {
                        if (ki > qi) {
                            scores[qi * T + ki] = Float.NEGATIVE_INFINITY; // causal mask
                            continue;
                        }
                        int kBase = (b * T + ki) * dModel + hOff;
                        float dot = 0f;
                        for (int d = 0; d < dHead; d++) {
                            dot += qd[qBase + d] * kd[kBase + d];
                        }
                        scores[qi * T + ki] = dot * scale;
                    }
                }

                // Softmax over last dim (key dim)
                float[] attn = new float[T * T];
                for (int qi = 0; qi < T; qi++) {
                    float max = Float.NEGATIVE_INFINITY;
                    for (int ki = 0; ki <= qi; ki++) {
                        if (scores[qi * T + ki] > max) max = scores[qi * T + ki];
                    }
                    float sum = 0f;
                    for (int ki = 0; ki <= qi; ki++) {
                        attn[qi * T + ki] = (float) Math.exp(scores[qi * T + ki] - max);
                        sum += attn[qi * T + ki];
                    }
                    for (int ki = 0; ki <= qi; ki++) {
                        attn[qi * T + ki] /= sum;
                    }
                }

                // Save for backward
                int awBase = (b * numHeads + h) * T * T;
                System.arraycopy(attn, 0, awd, awBase, T * T);

                // Weighted sum of V -> output
                for (int qi = 0; qi < T; qi++) {
                    int outBase = (b * T + qi) * dModel + hOff;
                    for (int ki = 0; ki <= qi; ki++) {
                        int vBase = (b * T + ki) * dModel + hOff;
                        float a = attn[qi * T + ki];
                        for (int d = 0; d < dHead; d++) {
                            od[outBase + d] += a * vd[vBase + d];
                        }
                    }
                }
            }
        }
        return out;
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        int B = lastB, T = lastT;

        // gradOutput: [B, T, dModel] -> flatten for outProj backward
        Tensor gradFlat = TensorOps.reshape(gradOutput, B * T, dModel);
        Tensor gradAttnOut = outProj.backward(gradFlat);   // [B*T, dModel]

        // Backprop through scaled dot-product attention
        float[] gaod = gradAttnOut.data();
        float[] awd  = lastAttnWeights.data();
        float[] qd   = lastQ.data();
        float[] kd   = lastK.data();
        float[] vd   = lastV.data();
        float scale  = (float) (1.0 / Math.sqrt(dHead));

        Tensor gradQ = new Tensor(B * T, dModel);
        Tensor gradK = new Tensor(B * T, dModel);
        Tensor gradV = new Tensor(B * T, dModel);
        float[] gqd = gradQ.data();
        float[] gkd = gradK.data();
        float[] gvd = gradV.data();

        for (int b = 0; b < B; b++) {
            for (int h = 0; h < numHeads; h++) {
                int hOff = h * dHead;
                int awBase = (b * numHeads + h) * T * T;

                for (int qi = 0; qi < T; qi++) {
                    // gradient through V weighted sum
                    int goBase = (b * T + qi) * dModel + hOff;
                    for (int ki = 0; ki <= qi; ki++) {
                        float a = awd[awBase + qi * T + ki];
                        int vBase = (b * T + ki) * dModel + hOff;
                        for (int d = 0; d < dHead; d++) {
                            // dL/dV[ki] += attn[qi,ki] * dL/dOut[qi]
                            gvd[vBase + d] += a * gaod[goBase + d];
                        }
                    }

                    // gradient through softmax -> scores -> Q,K
                    float[] dAttn = new float[T];
                    for (int ki = 0; ki <= qi; ki++) {
                        int vBase = (b * T + ki) * dModel + hOff;
                        for (int d = 0; d < dHead; d++) {
                            dAttn[ki] += gaod[goBase + d] * vd[vBase + d];
                        }
                    }

                    // softmax backward
                    float[] attnRow = new float[qi + 1];
                    float dot = 0f;
                    for (int ki = 0; ki <= qi; ki++) {
                        attnRow[ki] = awd[awBase + qi * T + ki];
                        dot += attnRow[ki] * dAttn[ki];
                    }
                    float[] dScores = new float[qi + 1];
                    for (int ki = 0; ki <= qi; ki++) {
                        dScores[ki] = attnRow[ki] * (dAttn[ki] - dot) * scale;
                    }

                    // gradient into Q and K
                    int qBase = (b * T + qi) * dModel + hOff;
                    for (int ki = 0; ki <= qi; ki++) {
                        int kBase = (b * T + ki) * dModel + hOff;
                        for (int d = 0; d < dHead; d++) {
                            gqd[qBase + d] += dScores[ki] * kd[kBase + d];
                            gkd[kBase + d] += dScores[ki] * qd[qBase + d];
                        }
                    }
                }
            }
        }

        // Backprop through Q/K/V projections
        Tensor gradInput  = new Tensor(B * T, dModel);
        float[] gi = gradInput.data();

        Tensor gQ = qProj.backward(gradQ);
        Tensor gK = kProj.backward(gradK);
        Tensor gV = vProj.backward(gradV);
        float[] gqd2 = gQ.data(), gkd2 = gK.data(), gvd2 = gV.data();
        for (int i = 0; i < gi.length; i++) {
            gi[i] = gqd2[i] + gkd2[i] + gvd2[i];
        }
        return TensorOps.reshape(gradInput, B, T, dModel);
    }

    @Override
    public List<Tensor> parameters() {
        List<Tensor> p = new ArrayList<>();
        p.addAll(qProj.parameters());
        p.addAll(kProj.parameters());
        p.addAll(vProj.parameters());
        p.addAll(outProj.parameters());
        return p;
    }
}
