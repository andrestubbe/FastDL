# FastDL — DOING.md
# Aktueller Arbeitsstand — Mini-LLM Transformer Stack + Performance

---

## Gesamtstruktur

```
fastdl/
├── tensor/
│   ├── Tensor.java                 OK vorhanden (heap-basiert)
│   └── OffHeapTensor.java          OK FERTIG — GC-frei, 32-byte aligned via fastmemory.Memory
├── ops/
│   ├── TensorOps.java              OK FERTIG (scalar Java, alle Ops)
│   └── VectorTensorOps.java        OK FERTIG — Java Vector API AVX2 matmul
├── layer/
│   ├── Layer.java / Dense / ReLU   OK vorhanden
│   ├── LayerNorm.java              OK FERTIG
│   └── GELU.java                   OK FERTIG
├── loss/
│   ├── MSELoss.java                OK vorhanden
│   └── CrossEntropyLoss.java       OK FERTIG
├── optim/
│   ├── SGD.java                    OK vorhanden
│   └── AdamW.java                  OK FERTIG
├── tokenizer/
│   ├── Tokenizer.java              OK FERTIG
│   └── CharTokenizer.java          OK FERTIG
├── data/
│   └── TextDataset.java            OK FERTIG
├── model/
│   ├── GPTConfig.java              OK FERTIG
│   ├── Embedding.java              OK FERTIG
│   ├── PositionalEncoding.java     OK FERTIG
│   ├── MultiHeadAttention.java     OK FERTIG
│   ├── FeedForward.java            OK FERTIG
│   ├── TransformerBlock.java       OK FERTIG
│   └── GPTModel.java               OK FERTIG
├── training/
│   ├── TrainerConfig.java          OK FERTIG
│   └── Trainer.java                OK FERTIG
├── generation/
│   └── Generator.java              OK FERTIG
├── backend/
│   ├── Backend.java                TODO — Perf Stufe 3 (Strategy Interface)
│   ├── CpuBackend.java             TODO — Perf Stufe 3
│   └── GpuBackend.java             TODO — Perf Stufe 4 (FastGPU Vulkan)
└── FastDL.java                     OK FERTIG (factory)
```

---

## Phase A — Transformer Stack OK ABGESCHLOSSEN

Alle Klassen implementiert, alle JavaDocs vollstaendig, Build gruen.

---

## Phase B — Performance-Beschleunigung

### Schritt 1 — Java 17 Vector API OK FERTIG

- pom.xml: --add-modules jdk.incubator.vector eingetragen
- ops/VectorTensorOps.java: AVX2 SIMD matmul + batchedMatmul, scalar fallback
- VectorTensorOpsTest.java: Test vorhanden

### Schritt 2 — FastMemory: GC-freie Tensor-Buffer OK FERTIG

- tensor/OffHeapTensor.java: direkt via fastmemory.Memory + fastpointer.Pointer
- pom.xml: FastCore 0.1.0, FastPointer 0.1.1, FastMemory 0.1.1 als Dependencies
- Tests: FastMemoryTest.java, FastMemoryIntegrationTest.java
- HINWEIS: Zwischenschicht-Facades (fastdl.memory.FastMemory/FastPointer) wurden entfernt
  OffHeapTensor nutzt fastmemory.Memory und fastpointer.Pointer direkt

### Schritt 3 — FastSIMD: Native AVX2 Matmul (JNI) TODO NAECHSTER SCHRITT

Was: FastSIMD um matmulAVX2(long ptrA, long ptrB, long ptrC, int M, int K, int N)
via _mm256_fmadd_ps erweitern. OffHeapTensor liefert bereits native Adressen
via fastpointer.Pointer.address() — die Infrastruktur ist bereit.

Voraussetzung: FastSIMD Repo braucht neue JNI-Methode (C++ Seite).

Dateien:
- (FastSIMD Repo) matmulAVX2 JNI-Methode in C++           TODO
- ops/SIMDTensorOps.java — JNI Bridge in FastDL            TODO
- ops/TensorOps.java — dispatch: SIMD > VectorAPI > scalar TODO

Speedup: ~4-8x. Pointer aus OffHeapTensor direkt in native matmul, null Copies.

### Schritt 4 — FastGPU: Vulkan Compute Matmul TODO SPAETER

Was: Dense.forward() + MultiHeadAttention auf GPU via FastGPU Vulkan Kernels.
FastGPU API: allocFloatBuffer() -> upload(float[]) -> dispatch(GLSL) -> download(float[]).

Dateien:
- backend/Backend.java — Strategy Interface (CPU/GPU austauschbar) TODO
- backend/GpuBackend.java — FastGPU Vulkan Bridge             TODO
- pom.xml — fastgpu 0.1.1 Dependency                          TODO

Speedup: ~10-25x auf matmul.

### Schritt 5 — FastSharedMemory: Multi-Process Distillation TODO ZUKUNFT

Teacher-Model in separatem Prozess, Student liest Logits via Zero-Copy SharedMemory.

---

## Demos

| Launcher                 | Demo                         | Status      |
|--------------------------|------------------------------|-------------|
| run-walkthrough.bat      | TinyStoriesWalkthroughDemo   | BEREIT      |
| run-tiny2.bat            | TinyStoriesTransformerDemo   | BEREIT      |
| run-tiny-big.bat         | TinyStoriesBigDemo           | BEREIT      |
| run-demo.bat             | LossSurfaceDemo              | vorhanden   |

---

## Offene Punkte

- [ ] Walkthrough-Demo: run-walkthrough.bat Maven exec Konfiguration pruefen
- [ ] TinyStories Erklaer-Demo (warum der Datensatz existiert) — noch nicht gebaut
- [ ] README.md API Quick Reference: Transformer factory Methoden erganzen
- [ ] Schritt 3 (FastSIMD) — naechste Performance-Stufe

---

## Build & Umgebung

- Workspace: C:\Users\andre\Documents\2026-08-17-Work-FastJava\FastDL
- Java 17, Maven 3.9.9: C:\Users\andre\tools\apache-maven-3.9.9
- Oekosystem: FastML (klassisches ML) | FastDL (Deep Learning) | FastGPU (GPU)
- Demo-Pattern: ANSI Konsolen-Style (grau/weiss)
- Datensatz: data/TinyStories-train.txt OK VORHANDEN (1.835 GB) — in .gitignore
