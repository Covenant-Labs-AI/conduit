# Conduit
Conduit strives to be an operating system for AI that provides type-safe, provider-agnostic pipelines for building, validating, training and deploying models across local and cloud compute.


---

## ✨ Core Concepts

### **Type-Safe AI Pipelines**
Conduit is built around strict, type-safe pipelines.

- Define request and response schemas using Python `@dataclass` types
- Conduit derives a formal schema from type hints and enforces it at runtime
- Payloads are validated automatically before and after model execution
- Generated type hints ensure models produce structurally correct outputs

This guarantees that every step in a pipeline operates on well-defined data.

---

### **Flexible Deployment**
Conduit decouples *what* you run from *where* and *how* it runs.

- Choose any supported **model**, **runtime**, and **compute provider**
- Conduit automatically:
  - Calculates GPU and memory requirements
  - Validates model–runtime–hardware compatibility
  - Provisions and manages deployments
  - Handles scaling and lifecycle management
  - Selects the most cost-efficient compute option

You can change runtimes or providers without modifying application logic.

---

### **AI-Native Workflow Blocks**
Pipelines are composed by chaining high-level, AI-aware building blocks.

- Each block represents a self-contained runtime capability
- Blocks can be combined to form complete inference and data-processing workflows

**Available blocks include:**
- `LMLiteBlock` — LLM inference runtime (streaming + batching)
- `HttpGetBlock`
- `FileSystemWriteBlock`
- `Sqlite3Block`
- `SystemCommandBlock`
- …with more blocks planned

Blocks are designed to be composable, observable, and production-safe.

---

### **Runtimes**
A **Runtime** is any system capable of loading and executing a model.

- Runtimes define *how* inference is performed:
  - Execution engine
  - Batching and streaming behavior
  - Memory management
  - Concurrency control
- Conduit abstracts runtimes behind a unified interface, allowing runtime changes without code changes

**Currently supported runtimes:**
- **LmLite** *(experimental)*
  - Conduit’s native LLM runtime
  - Supports multiple models loaded concurrently
  - Async batching support
  - Per-model concurrency limits

**Planned runtimes:**
- **vLLM** *(coming next)*
  - High-throughput LLM inference
  - Optimized for large-scale, multi-request workloads
- **TensorRT** *(planned)*
  - NVIDIA-optimized inference runtime
  - CUDA-based execution
  - Best suited for production, fixed-shape inference workloads

---

### **Compute Providers**
A **Compute Provider** is *where* your runtime is deployed. Responsible for  provisioning and managing the lifecycle of containerized runtime environments, either locally or on remote infrastructure.

Conduit abstracts all providers behind a common adapter interface. Each provider implements the same lifecycle semantics (create, start, restart, stop, terminate), allowing Conduit to remain provider-agnostic while supporting heterogeneous execution environments. Provider-specific APIs, responses, and behaviors are normalized internally via adapters.

This design allows you to switch between local and cloud GPUs without changing application logic.

---

#### **Supported Compute Providers**

##### **Local NVIDIA (Docker)**
Runs Conduit on a local machine with NVIDIA GPUs using Docker.

This provider uses Docker as the container runtime and requires the NVIDIA Container Toolkit to expose GPUs to containers.

**Prerequisites:**
- NVIDIA GPU with compatible drivers installed
- Docker
- NVIDIA Container Toolkit

```bash
# Add docker group (if missing) and add current user
sudo groupadd docker 2>/dev/null || true
sudo usermod -aG docker "$USER"

# Add NVIDIA Container Toolkit repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg

curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

# Install NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use the NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker
sudo systemctl restart docker
```

##### **Runpod**
Runs Conduit on cloud-hosted GPUs via Runpod.

This provider provisions on-demand GPU instances and manages container lifecycle remotely.

**Prerequisites:**
- A Runpod account
- An API key configured in the environment

```bash
export RUNPOD_API_KEY="your_api_key_here"
```


### **Compute Offerings**
A **Compute Offering** is Conduit’s normalized representation of provider compute capacity.

- All non-local compute providers are abstracted into one or more compute offerings
- Each offering defines:
  - Available GPU memory
  - Hourly price
  - Maximum availability
  - Provider metadata

This abstraction allows Conduit to reason about heterogeneous compute providers using a single model.

---

### **Provisioning & Compute Selection**
During provisioning, Conduit evaluates all available compute offerings and selects the best option for a deployment.

- Conduit runs an internal `calculate_best_compute_offering` step that:
  - Evaluates compatible compute offerings
  - Matches them against model requirements and deployment constraints
  - Selects the most suitable offering

**Current selection criteria:**
- Best price per GB of GPU memory (VRAM)

**Planned enhancements:**
- GPU topology awareness
- Compute performance metrics (e.g. FLOPs / TFLOPs)
- Multi-GPU and multi-node placement optimization

**Why Compute Offerings Matter**
- Enables provider-agnostic deployments
- Allows automatic cost optimization
- Provides a foundation for topology-aware scheduling

---

### **MDL (Model Data Language)**
MDL is Conduit’s schema format for describing structured input and output data.

- Generated directly from Python `@dataclass` type hints
- Defines exact data shapes using primitives, `List`, `Dict`, `Optional`, and `Union`
- Expands nested dataclasses into explicit dictionary structures
- Disallows untyped `Any` and explicit `None`
- Enforces strict, schema-correct JSON generation from models


## 📦 Installation

### Requirements
- **Python 3.12.6** or newer

### Install from GitHub (latest)

```bash
pip install "git+https://github.com/Covenant-Labs-AI/conduit.git@main"

```
## 🔥 LMLite — AI Inference Runtime

Conduit integrates seamlessly with **LMLite** — our multi-model/multi-GPU batching engine (still under development!).  
It handles:
- Efficient batch processing
- Supports multiple models loaded concurrently
- Provides per-model concurrency control

> Add as many models and replicas as you want — Conduit will calculate model constraints and throw an error if something doesn't fit.

---

```python
# Transforms directory listings into structured data using an LLM
model_id="Qwen/Qwen3-4B-Instruct-2507-FP8"

@dataclass
class Command:
    shell_command: str

@dataclass
class DirectoryListing:
    listing: str

@dataclass
class File:
    name: str
    bytes: int
    date_modified: str

@dataclass
class Files:
    files: List[File]

command_block = SystemCommandBlock(Command, timeout_seconds=5)

lm_lite_block = LMLiteBlock(
    models=[
        LmLiteModelConfig(
            model_id,
            max_model_len=5000,
            max_model_concurrency=1,
        ),
    ],
    compute_provider=ComputeProvider.LOCAL
)

command_op = command_block(Command(shell_command="ls -al")) 
if command_op.success:
    directory_input = DirectoryListing(listing=command_op.stdout)
    result = lm_lite_block(
        model_id=model_id, input=directory_input, output=Files
    )
    print(result) # List[Files]

```

## ✅ TODOs

- [ ] Support **vLLM** as a runtime (e.g. `VllmBlock` alongside `LMLiteBlock`)
- [ ] Expand supported compute providers:
  - [ ] **AWS** (EKS / ECS / EC2)
  - [ ] **GCP** (GKE / Vertex / Compute Engine)
  - [ ] **Azure** (AKS / Azure ML)
  - [ ] **Lambda Labs**
  - [ ] **Modal**
  - [ ] **CoreWeave**
