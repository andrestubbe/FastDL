package fastdl.model;

import fastdl.layer.Dense;
import fastdl.layer.GELU;
import fastdl.layer.Layer;
import fastdl.ops.TensorOps;
import fastdl.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Transformer FeedForward block.
 *
 * Structure: Dense(dModel -> 4*dModel) -> GELU -> Dense(4*dModel -> dModel)
 *
 * Input/output shape: [batch, seqLen, dModel]
 * Internally flattened to [batch*seqLen, dModel] for the Dense layers.
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
