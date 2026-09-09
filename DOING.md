# FastDL — DOING.md
# Aktueller Arbeitsstand — Mini-LLM Transformer Stack + Performance

---

## Ziel
Vollständiger Mini-LLM Transformer Stack in FastDL (Java 17, Maven),
plus schrittweise Performance-Beschleunigung durch das FastJava Ecosystem.

---

## Gesamtstruktur

```
fastdl/
├── tensor/Tensor.java              ✅ vorhanden
├── ops/
│   ├── TensorOps.java              ✅ FERTIG (scalar Java)
│   └── VectorTensorOps.java        ⬜ IN ARBEIT — Java Vector API (AVX2, kein JNI)
├── layer/
│   ├── Layer.java                  ✅ vorhanden
│   ├── Dense.java                  ✅ vorhanden
│   ├── ReLU.java                   ✅ vorhanden
│   ├── LayerNorm.java              ✅ FERTIG
│   └── GELU.java                   ✅ FERTIG
├── loss/
│   ├── MSELoss.java                ✅ vorhanden
│   └── CrossEntropyLoss.java       ✅ FERTIG
├── optim/
│   ├── SGD.java                    ✅ vorhanden
│   └── AdamW.java                  ✅ FERTIG
├── tokenizer/
│   ├── Tokenizer.java              ✅ FERTIG
│   └── CharTokenizer.java          ✅ FERTIG
├── data/
│   └── TextDataset.java            ✅ FERTIG
├── model/
│   ├── GPTConfig.java              ✅ FERTIG
│   ├── Embedding.java              ✅ FERTIG
│   ├── PositionalEncoding.java     ✅ FERTIG
│   ├── MultiHeadAttention.java     ✅ FERTIG
│   ├── FeedForward.java            ✅ FERTIG
│   ├── TransformerBlock.java       ✅ FERTIG
│   └── GPTModel.java               ✅ FERTIG
├── training/
│   ├── TrainerConfig.java          ✅ FERTIG
│   └── Trainer.java                ✅ FERTIG
├── generation/
│   └── Generator.java              ✅ FERTIG
├── backend/
│   ├── Backend.java                ⬜ TODO — Perf Stufe 2
│   ├── CpuBackend.java             ⬜ TODO — Perf Stufe 2
│   └── GpuBackend.java             ⬜ TODO — Perf Stufe 4
└── FastDL.java                     ✅ FERTIG (factory)
```

---

## Phase A — Transformer Stack (Basis) ✅ ABGESCHLOSSEN

Alle Klassen implementiert, Build grün, Demo läuft.

---

## Phase B — Performance-Beschleunigung (NEU — IN ARBEIT)

Quelle: Analyse des FastJava Ecosystems (FastSIMD, FastMemory, FastGPU, FastPointer, FastSharedMemory).

### Schritt 1 — Java 17 Vector API in TensorOps ⬜ IN ARBEIT

**Was:** Java 17 `jdk.incubator.vector` — AVX2-Vektoren direkt aus Java, ohne JNI, ohne
nativen Build. Ersetzt die innerste matmul-Schleife durch 8-float-pro-Cycle SIMD.

**Dateien:**
| Datei | Status |
|---|---|
| `pom.xml` — `--add-modules jdk.incubator.vector` | ⬜ TODO |
| `ops/VectorTensorOps.java` — SIMD matmul + dot | ⬜ TODO |
| `ops/TensorOps.java` — dispatch auf VectorTensorOps | ⬜ TODO |

**Speedup:** ~3–5x auf matmul ohne irgendeine Abhängigkeit.

---

### Schritt 2 — FastMemory: GC-freie Tensor-Buffer

**Was:** `Tensor`'s interne `float[]` durch 32-byte aligned off-heap Buffer via FastMemory
ersetzen. Kein GC-Jitter während Training bei großen Batches.

**Dateien:**
| Datei | Status |
|---|---|
| `tensor/OffHeapTensor.java` — FastMemory + FastPointer | ⬜ TODO |
| `pom.xml` — FastMemory 0.1.1, FastPointer 0.1.1 Dep | ⬜ TODO |

**Speedup:** Keine GC-Pausen. Kritisch bei batchSize ≥ 8 oder seqLen ≥ 256.

---

### Schritt 3 — FastSIMD: Native AVX2 Matmul (JNI Extension)

**Was:** FastSIMD erweitern um `matmulAVX2(long ptrA, long ptrB, long ptrC, int M, int K, int N)`
via `_mm256_fmadd_ps`. Erfordert C++ Extension in FastSIMD.

**Dateien:**
| Datei | Status |
|---|---|
| (FastSIMD Repo) — neuer JNI-Aufruf `matmulAVX2` | ⬜ TODO (anderer Repo) |
| `ops/SIMDTensorOps.java` — JNI Bridge | ⬜ TODO |
| `ops/TensorOps.java` — dispatch auf SIMDTensorOps | ⬜ TODO |

**Speedup:** ~4–8x auf matmul. AVX2 FMA = 8 FP32 Multiply-Accumulate pro Cycle.

---

### Schritt 4 — FastGPU: Vulkan Compute Matmul

**Was:** `Dense.forward()` + `MultiHeadAttention` auf GPU via FastGPU Vulkan Compute Kernels.
FastGPU API: `allocFloatBuffer()` → `upload(float[])` → `dispatch(GLSL kernel)` → `download(float[])`.

**GLSL Kernel:**
```glsl
#version 450
layout(local_size_x = 16, local_size_y = 16) in;
layout(set=0, binding=0) readonly  buffer A { float a[]; };
layout(set=0, binding=1) readonly  buffer B { float b[]; };
layout(set=0, binding=2) writeonly buffer C { float c[]; };
layout(push_constant) uniform PC { int M; int K; int N; } pc;
void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    if (row >= pc.M || col >= pc.N) return;
    float sum = 0.0;
    for (int k = 0; k < pc.K; k++) sum += a[row * pc.K + k] * b[k * pc.N + col];
    c[row * pc.N + col] = sum;
}
```

**Dateien:**
| Datei | Status |
|---|---|
| `backend/Backend.java` — Strategy Interface | ⬜ TODO |
| `backend/GpuBackend.java` — FastGPU Vulkan Bridge | ⬜ TODO |
| `pom.xml` — fastgpu 0.1.1 Dependency | ⬜ TODO |

**Speedup:** ~10–25x auf matmul. GPU hat 1000+ Shader Units parallel.

---

### Schritt 5 — FastSharedMemory: Multi-Process Distillation

**Was:** Teacher-Model in separatem Prozess, Student liest Logits via Zero-Copy SharedMemory.
Ermöglicht Knowledge Distillation ohne Serialisierungs-Overhead.

**Dateien:**
| Datei | Status |
|---|---|
| `fastdl/distill/DistillTrainer.java` | ⬜ TODO (Zukunft) |

---

## Designentscheidungen

| Entscheidung | Gewählt | Grund |
|---|---|---|
| Tokenizer | CharTokenizer | Einfach, kein OOV, ~100 Tokens |
| Modell | GPT (Pre-LN) | Stabiler als Post-LN für kleine Modelle |
| Optimizer | AdamW | Standard für Transformer |
| Backend (initial) | Pure JVM scalar | Kein JNI nötig zum Start |
| dModel | 128 (demo), 256 (standard) | Schnell trainierbar auf CPU |
| seqLen | 64–128 | TinyStories Stories sind kurz |
| CRLF | alle .bat Dateien | Windows Requirement |

---

## Datensatz

- `data/TinyStories-train.txt` ✅ VORHANDEN (1.835 GB, echte HuggingFace Daten)
- `data/tinystories.txt` — kleiner Platzhalter (Backup für schnelle Tests)
- `.gitignore` schließt `data/*.txt` aus — nicht ins Repo committen!

---

## Anleitung zum Weitermachen

1. Diese Datei lesen
2. Nächste `⬜ TODO` Zeile in "Phase B" suchen
3. Aktuell: **Schritt 1 — Java Vector API** ist der nächste Schritt
4. Nach jeder Änderung: `mvn clean install -DskipTests -q` aus FastDL-Root
5. Dann Demo testen: `run-tiny2.bat`

---

## Build & Umgebung

- Workspace: `C:\Users\andre\Documents\2026-08-17-Work-FastJava\FastDL`
- Java 17, Maven 3.9.9: `C:\Users\andre\tools\apache-maven-3.9.9`
- Ökosystem: FastML (klassisches ML) | FastDL (Deep Learning) | FastGPU (GPU)
- Demo-Pattern: ANSI Konsolen-Style (grau/weiss)
