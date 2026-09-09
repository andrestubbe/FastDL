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
- [Key Features](#key-features)
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
- **Mini LLM / Transformer Foundations** — including attention, feedforward blocks, embeddings, causal generation, and training loops.
- **TinyStories-Ready Workflow** — local datasets, tokenizer pipelines, and model training code built for small-language-model experimentation.

---

## Key Features

- **🧱 Dense & Multidimensional Tensors** — Zero-copy flat buffers, strides, automatic gradient tracking (`grad`).
- **🧠 Neural Layers & Modular Sequentials** — `Dense`, `ReLU`, `LayerNorm`, `GELU`, and composable model blocks.
- **⚡ Optimizers with Momentum** — Stochastic Gradient Descent and AdamW for transformer-style training.
- **📉 Non-Convex Optimization & Loss Surfaces** — Built-in loss metrics (`MSELoss`, `CrossEntropyLoss`) and minima exploration.
- **🤖 Mini Transformer / GPT-Style Stack** — token embeddings, positional encoding, causal attention, feedforward blocks, and text generation.
- **📚 TinyStories Demo Path** — end-to-end local text-model demo workflow built around TinyStories-style corpora.

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

- Non-Convex Loss Surface & Minima Valley — [examples/Demo/src/main/java/fastdl/demo/LossSurfaceDemo.java](examples/Demo/src/main/java/fastdl/demo/LossSurfaceDemo.java)
- TinyStories Preview — [examples/TinyStoriesDemo/src/main/java/fastdl/demo/TinyStoriesMiniDemo.java](examples/TinyStoriesDemo/src/main/java/fastdl/demo/TinyStoriesMiniDemo.java)
- TinyStories Transformer Demo — [examples/TinyStoriesDemo/src/main/java/fastdl/demo/TinyStoriesTransformerDemo.java](examples/TinyStoriesDemo/src/main/java/fastdl/demo/TinyStoriesTransformerDemo.java)
- TinyStories Big-Data Runner — [run-tiny-big.bat](run-tiny-big.bat) and [examples/TinyStoriesDemo/src/main/java/fastdl/demo/TinyStoriesTransformerDemo.java](examples/TinyStoriesDemo/src/main/java/fastdl/demo/TinyStoriesTransformerDemo.java)
- MLP Boundary Visualization — [examples/Demo/src/main/java/fastdl/demo/MLPBoundaryDemo.java](examples/Demo/src/main/java/fastdl/demo/MLPBoundaryDemo.java)
- Autoencoder / Representation Learning — [examples/Demo/src/main/java/fastdl/demo/AutoencoderDemo.java](examples/Demo/src/main/java/fastdl/demo/AutoencoderDemo.java)

### Root launchers

The repo also includes simple Windows batch launchers for the demo flow:

- `run-demo.bat` — generic project demo entrypoint
- `run-demo-autoencoder.bat` — autoencoder demo
- `run-demo-mlp.bat` — MLP demo
- `run-tiny-quick.bat` — Stage 1: TinyStories quick preview / smoke run
- `run-tiny-transformer.bat` — Stage 2: TinyStories transformer smoke run
- `run-tiny-big.bat` — Stage 3: long TinyStories big-data run
- legacy aliases: `run-tiny.bat` and `run-tiny2.bat` delegate to the new stage launchers

These launchers are designed to compile the root project, install the local artifact, and then execute the relevant example from the Java demo modules.

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
