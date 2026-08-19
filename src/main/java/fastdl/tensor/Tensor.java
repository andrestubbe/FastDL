package fastdl.tensor;

import java.util.Arrays;
import java.util.Random;

/**
 * Multidimensional Tensor backing FastDL operations.
 */
public final class Tensor {

    private final float[] data;
    private final float[] grad;
    private final int[] shape;
    private final int[] strides;

    public Tensor(int... shape) {
        this.shape = shape != null && shape.length > 0 ? shape.clone() : new int[]{1};
        this.strides = computeStrides(this.shape);
        int total = computeSize(this.shape);
        this.data = new float[total];
        this.grad = new float[total];
    }

    public Tensor(float[] data, int... shape) {
        this.shape = shape != null && shape.length > 0 ? shape.clone() : new int[]{data.length};
        this.strides = computeStrides(this.shape);
        this.data = data != null ? data.clone() : new float[computeSize(this.shape)];
        this.grad = new float[this.data.length];
    }

    public int[] shape() {
        return shape.clone();
    }

    public int size() {
        return data.length;
    }

    public float[] data() {
        return data;
    }

    public float[] grad() {
        return grad;
    }

    public float get(int... indices) {
        return data[offset(indices)];
    }

    public void set(float val, int... indices) {
        data[offset(indices)] = val;
    }

    public float getGrad(int... indices) {
        return grad[offset(indices)];
    }

    public void setGrad(float val, int... indices) {
        grad[offset(indices)] = val;
    }

    public void addGrad(float val, int... indices) {
        grad[offset(indices)] += val;
    }

    public void zeroGrad() {
        Arrays.fill(grad, 0f);
    }

    public static Tensor randn(int... shape) {
        Tensor t = new Tensor(shape);
        Random r = new Random();
        for (int i = 0; i < t.data.length; i++) {
            t.data[i] = (float) (r.nextGaussian() * 0.1);
        }
        return t;
    }

    public static Tensor zeros(int... shape) {
        return new Tensor(shape);
    }

    public static Tensor of(float... values) {
        return new Tensor(values, values.length);
    }

    private int offset(int... indices) {
        int off = 0;
        for (int i = 0; i < indices.length; i++) {
            off += indices[i] * strides[i];
        }
        return off;
    }

    private static int[] computeStrides(int[] shape) {
        int[] s = new int[shape.length];
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            s[i] = stride;
            stride *= shape[i];
        }
        return s;
    }

    private static int computeSize(int[] shape) {
        int size = 1;
        for (int dim : shape) size *= dim;
        return size;
    }

    @Override
    public String toString() {
        return "Tensor(shape=" + Arrays.toString(shape) + ", data=" + (data.length <= 8 ? Arrays.toString(data) : "[" + data[0] + ", ..., " + data[data.length - 1] + "]") + ")";
    }
}
