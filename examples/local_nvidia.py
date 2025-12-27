import socket
from dataclasses import dataclass
from pathlib import Path
import os
import platform
from typing import List
from pathlib import Path
from conduit.compute_provider.local import LocalNetworkBinding
from conduit.runtime import LMLiteBlock
from conduit import LmLiteModelConfig, ComputeProvider
from conduit.blocks import FileSystemReadBlock, SystemCommandBlock

# -----------------------------------------------------------------------------
# Demo: Local NVIDIA (GPU) compatibility via Conduit + LMLite
#
# What this script does:
# - Starts an LMLite deployment using the LOCAL compute provider
#   (runs on your machine; uses NVIDIA GPU + CUDA if available).
# - Verifies the full Conduit workflow end-to-end:
#     • model startup and readiness
#     • local inference
#     • shell command execution
#     • filesystem reads
#     • structured input/output with typed dataclasses
#
# Prerequisites (Docker + NVIDIA runtime setup):
#
# ## presteps added docker to group
# sudo groupadd docker 2>/dev/null || true
# sudo usermod -aG docker "$USER"
#
#
# # 1) Add NVIDIA Container Toolkit repo
# curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
#   | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg
#
# curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
#   | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit.gpg] https://#g' \
#   | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
#
# # 2) Install toolkit
# sudo apt-get update
# sudo apt-get install -y nvidia-container-toolki
#
#
# # Configure Docker to use the NVIDIA runtime
# sudo nvidia-ctk runtime configure --runtime=docker
#
# # Restart Docker daemon
# sudo systemctl restart docker
#
# Purpose:
# - Acts as a lightweight “smoke test” for local GPU setup and Conduit + LMLite
#   integration, mirroring the same deployment pattern used in cloud runs.
# -----------------------------------------------------------------------------


def test_local_nvidia(model_id: str = "Qwen/Qwen3-4B-Instruct-2507-FP8"):
    """
    Local NVIDIA test harness:
    - Collect basic system info
    - Ask LMLite model for an `ls -l` command
    - Execute it
    - Parse listing into file metadata
    - For each .md file: read, summarize, and send a desktop notification
    """

    # LMLite deployment block: defines what to run, where to run it, and how to place/scale it.
    lm_lite_block = LMLiteBlock(
        models=[
            # List as many models as you want. Conduit/LMLite will validate feasibility
            # (VRAM/compute) and error if the request can’t be satisfied.
            LmLiteModelConfig(
                model_id,  # Hugging Face model id
                max_model_len=5000,  # Configure runtime for this max context length
                max_model_concurrency=1,  # Per-replica concurrency / request pool size
            ),
        ],
        # Compute provider (where the deployment runs)
        compute_provider=ComputeProvider.LOCAL,
        # --- Runpod cloud selection quirk ---
        # Runpod has multiple cloud pools (e.g., ENTERPRISE/secure vs COMMUNITY).
        #
        # - constraints=[DeploymentConstraint.ENTERPRISE] filters placement to enterprise-eligible capacity.
        # - Switching to COMMUNITY on Runpod requires a provider override:
        #     compute_provider_config_overrides={"cloudType": "COMMUNITY"}
        #   And you must REMOVE the ENTERPRISE constraint, otherwise you’ll filter out community capacity.
        #
        # Example (community):
        #   constraints=[]
        #   compute_provider_config_overrides={"cloudType": "COMMUNITY"}
        #
        # compute_provider_config_overrides={"cloudType": "COMMUNITY"},  # Runpod-only
        # Placement / compliance constraints (scheduler-side filtering)
        # constraints=[
        #    DeploymentConstraint.ENTERPRISE,
        #    DeploymentConstraint.SINGLE_DEVICE,
        # ],  # SOC2 compliant T3/T4 datacenters only
        # --- Hardware selection (when applicable) ---
        # Rule: If compute_provider is LOCAL, `num_gpus` and `gpu` do nothing (ignored).
        # For non-local providers, hardware is either auto-selected by Conduit or controlled via
        # provider-specific mechanisms (not via LOCAL-style pinning).
        #
        # num_gpus=2,
        # gpu=GPUS.L4,
        # Replica count (LMLite does round-robin load balancing across replicas)
        replicas=1,
    )

    @dataclass
    class Command:
        shell_command: str

    @dataclass(frozen=True)
    class BasicSystemInfo:
        hostname: str
        user: str | None
        os: str
        os_release: str
        os_version: str
        machine: str
        architecture: str
        processor: str
        python_version: str
        home: str
        cwd: str

        @classmethod
        def from_os(cls) -> "BasicSystemInfo":
            return cls(
                hostname=socket.gethostname(),
                user=os.getenv("USERNAME") or os.getenv("USER"),
                os=platform.system(),
                os_release=platform.release(),
                os_version=platform.version(),
                machine=platform.machine(),
                architecture=platform.architecture()[0],
                processor=platform.processor(),
                python_version=platform.python_version(),
                home=str(Path.home()),
                cwd=os.getcwd(),
            )

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

    @dataclass
    class ReadmeFileContent:
        content: str

    @dataclass
    class ProjectSummary:
        summary: str
        random_app_idea_with_conduit: str

    info = BasicSystemInfo.from_os()
    command_block = SystemCommandBlock(Command, timeout_seconds=5)

    if not lm_lite_block.ready:
        print("not ready")
        return

    # Ask the model for a shell command to list the current directory (long format).
    result = lm_lite_block(
        model_id=model_id,
        input=info,
        output=Command,
        guidance="list current directroy long format",
    )

    op = command_block(result)
    if not op.success:
        return

    # Parse the directory listing into structured file metadata.
    directory_input = DirectoryListing(listing=op.stdout)
    result = lm_lite_block(model_id=model_id, input=directory_input, output=Files)

    for file in result.files:
        if not file.name.endswith(".md"):
            continue

        readme = FileSystemReadBlock(path=Path(file.name))
        data = readme()

        content = ReadmeFileContent(content=data.data)
        project_sum = lm_lite_block(
            model_id=model_id,
            input=content,
            output=ProjectSummary,
        )

        cmd_result = command_block(
            Command(
                shell_command=(
                    'notify-send -u critical -t 5000 "Conduit" '
                    f'"{project_sum.random_app_idea_with_conduit}"'
                )
            )
        )

        if not cmd_result.success:
            print(f"NOTIFY-SEND failed: {project_sum.random_app_idea_with_conduit}")


if __name__ == "__main__":
    test_local_nvidia()
