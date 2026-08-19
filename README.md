# FastDL 0.1.0 [ALPHA-2026-08] — Deep Learning Engine, Tensor Computing & Neural Backprop for Java

[![Status](https://img.shields.io/badge/status-0.1.0-brightgreen.svg)](https://github.com/andrestubbe/FastDL/releases/tag/0.1.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Java](https://img.shields.io/badge/Java-17+-blue.svg)](https://www.java.com)
[![Platform](https://img.shields.io/badge/Platform-Cross--Platform-lightgrey.svg)]()
[![JitPack](https://img.shields.io/badge/JitPack-ready-green.svg)](https://jitpack.io/#andrestubbe/FastDL)

---

**⚡ Lightweight Deep Learning, high-performance tensor operations, gradient descent optimizers, and neural layers for the FastJava ecosystem.**

**FastDL** is the Deep Learning engine of the **FastJava** ecosystem. While **FastML** focuses on classical, deterministic pattern models with hand-crafted features (Centroids, KNN, SVM), **FastDL** provides the neural substrate: Multidimensional Tensors, Automatic Differentiation / Backpropagation, Neural Layers (`Dense`, `ReLU`, `Conv`), Optimizers (`SGD`, `Momentum`, `Adam`), and Loss surfaces.

```java
// Quick Start — Example
import fastdl.FastDL;
import fastdl.FastDL.Sequential;
import fastdl.loss.MSELoss;
import fastdl.optim.SGD;
import fastdl.tensor.Tensor;

public class Demo {
    public static void main(String[] args) {
        // 1. Define Multi-Layer Perceptron (MLP)
        Sequential net = FastDL.sequential(
            FastDL.dense(2, 8),
            FastDL.relu(),
            FastDL.dense(8, 1)
        );

        // 2. Setup Optimizer and Loss
        SGD optimizer = FastDL.sgd(net.parameters(), 0.01f, 0.9f);
        MSELoss criterion = FastDL.mse();

        // 3. Forward Pass & Training Step
        Tensor x = FastDL.tensor(new float[]{0.5f, -0.2f}, 1, 2);
        Tensor target = FastDL.tensor(new float[]{1.0f}, 1, 1);

        optimizer.zeroGrad();
        Tensor pred = net.forward(x);
        float loss = criterion.forward(pred, target);
        
        net.backward(criterion.backward(pred, target));
        optimizer.step();

        System.out.printf("Loss: %.4f | Output: %s%n", loss, pred);
    }
}
```

---

## Table of Contents

- [Why FastDL?](#why-fastdl)
- [FastML vs. FastDL](#fastml-vs-fastdl)
- [Key Features](#key-features)
- [Architecture & FastJava Ecosystem](#architecture--fastjava-ecosystem)
- [API Quick Reference](#api-quick-reference)
- [Installation](#installation)
- [Technical Examples & Hero Demos](#technical-examples--hero-demos)
- [Documentation](#documentation)
- [Platform Support](#platform-support)
- [License](#license)
- [Related Projects](#related-projects)

---

## Why FastDL?

Standard Deep Learning frameworks in the Java ecosystem (like DL4J) suffer from bloated dependencies, complex native bridges, and heavy memory footprints.

**FastDL** delivers:

- **100% Pure JVM Core with Optional Native SIMD/GPU Acceleration** — Instant startup, zero setup friction.
- **Microsecond Tensor Operations** — Cache-friendly flat arrays with stride-based multidimensional indexing.
- **Zero Framework Bloat** — Minimalist, PyTorch-like layer and optimizer APIs designed specifically for FastJava.

---

## FastML vs. FastDL

| Property | FastML | FastDL |
|---|---|---|
| **Approach** | Classical ML (Centroids, KNN, Trees) | Deep Neural Networks (MLP, CNN, Autoencoders) |
| **Feature Extraction** | Handcrafted / Extracted by Developer | Learned end-to-end by the Network |
| **Data Representation** | `VectorPattern`, `RasterPattern` | Multidimensional `Tensor` with gradients |
| **Optimization** | Direct Analytical / Nearest Distance | Backpropagation + Gradient Descent (`SGD`, `Adam`) |
| **Primary Substrate** | CPU / FastMath | CPU / FastGPU acceleration |

---

## Key Features

- **🧱 Dense & Multidimensional Tensors** — Zero-copy flat buffers, strides, automatic gradient tracking (`grad`).
- **🧠 Neural Layers & Modular Sequentials** — `Dense`, `ReLU`, custom composable activation layers.
- **⚡ Optimizers with Momentum** — Stochastic Gradient Descent with velocity momentum tracking.
- **📉 Non-Convex Optimization & Loss Surfaces** — Built-in loss metrics (`MSELoss`) and minima exploration.

---

## Architecture & FastJava Ecosystem

```text
                    FastAI (High-Level AI & Agents)
                         │
                   ┌─────┴─────────────┐
                   ▼                   ▼
             FastModel               FastDL (This Library)
          (LLMs & Embeddings)      (Deep Learning & Tensors)
                                       │
                         ┌─────────────┼─────────────┐
                         ▼             ▼             ▼
                      Tensors       Layers      Optimizers
                     (Strides,      (Dense,       (SGD,
                      Grads)         ReLU)       Momentum)
```

---

## API Quick Reference

| Method | Description |
|---|---|
| `FastDL.tensor(shape...)` | Allocates a zero-initialized tensor. |
| `FastDL.dense(in, out)` | Creates a fully-connected linear layer. |
| `FastDL.relu()` | Rectified Linear Unit activation layer. |
| `FastDL.sequential(layers...)` | Chains layers into an executable network container. |
| `FastDL.sgd(params, lr, momentum)` | Creates an SGD optimizer with momentum. |
| `FastDL.mse()` | Mean Squared Error loss calculator. |

---

## Installation

### Option 1: Maven (Recommended)

Add the JitPack repository and dependency to your `pom.xml`:

```xml
<repositories>
    <repository>
        <id>jitpack.io</id>
        <url>https://jitpack.io</url>
    </repository>
</repositories>

<dependencies>
    <dependency>
        <groupId>com.github.andrestubbe</groupId>
        <artifactId>FastDL</artifactId>
        <version>0.1.0</version>
    </dependency>
</dependencies>
```

### Option 2: Gradle (via JitPack)

```groovy
repositories {
    maven { url 'https://jitpack.io' }
}

dependencies {
    implementation 'com.github.andrestubbe:FastDL:0.1.0'
}
```

---

## Technical Examples & Hero Demos

| Case | Java Example | Description |
|---|---|---|
| Non-Convex Loss Surface & Minima Valley | [LossSurfaceDemo.java](examples/Demo/src/main/java/fastdl/demo/LossSurfaceDemo.java) | Real-time interactive simulation of gradient descent and momentum balls escaping local minima |

---

## Platform Support

| Platform | Status |
|---|---|
| **Windows 10/11** | ✅ Fully Supported |
| **Linux** | ✅ Fully Supported |
| **macOS** | ✅ Fully Supported |

---

## License

MIT License — See [LICENSE](LICENSE) for details.

---

## Related Projects

- [FastML](https://github.com/andrestubbe/FastML) — Classical Machine Learning and deterministic pattern recognition
- [FastAI](https://github.com/andrestubbe/FastAI) — High-level unified AI and reasoning substrate
- [FastModel](https://github.com/andrestubbe/FastModel) — Local GGUF/ONNX model runtimes
- [FastGPU](https://github.com/andrestubbe/FastGPU) — Vulkan and GPU compute acceleration

---

**Part of the FastJava Ecosystem** — *Making the JVM faster. Small package. Maximum speed. Zero bloat. 🚀*
