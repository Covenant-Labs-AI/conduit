import subprocess
from dataclasses import dataclass
from typing import (
    Type,
    Protocol,
    runtime_checkable,
)


from dataclasses import dataclass, is_dataclass, asdict, fields
from typing import Type

from .block_types import Block, I, O


@runtime_checkable
class SupportsShellCommand(Protocol):
    shell_command: str


@dataclass
class SystemCommandOperation:
    success: bool
    command: str
    return_code: int | None = None
    stdout: str | None = None
    stderr: str | None = None
    reason: str | None = None
    syntax_ok: bool | None = None


class SystemCommandBlock[Input: SupportsShellCommand](
    Block[Input, SystemCommandOperation]
):
    """
    Validates shell syntax, then executes the command.

    Syntax validation uses: <shell> -n -c "<cmd>"
    Execution uses:         <shell> -c "<cmd>"
    """

    def __init__(
        self,
        input: Type[Input],
        shell: str = "bash",
        timeout_seconds: float | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ):
        super().__init__(input, SystemCommandOperation)
        self.shell = shell
        self.timeout_seconds = timeout_seconds
        self.cwd = cwd
        self.env = env

    def forward(self, data: Input) -> SystemCommandOperation:
        if not hasattr(data, "shell_command"):
            return SystemCommandOperation(
                success=False,
                command="",
                reason="SystemCommandBlock input dataclass must have a 'shell_command' field",
                syntax_ok=False,
            )

        cmd = data.shell_command

        # 1) Syntax validation (no execution)
        try:
            syntax = subprocess.run(
                [self.shell, "-n", "-c", cmd],
                capture_output=True,
                text=True,
                cwd=self.cwd,
                env=self.env,
                timeout=self.timeout_seconds,
            )
        except FileNotFoundError:
            return SystemCommandOperation(
                success=False,
                command=cmd,
                reason=f"Shell not found: {self.shell}",
                syntax_ok=False,
            )
        except subprocess.TimeoutExpired:
            return SystemCommandOperation(
                success=False,
                command=cmd,
                reason="Syntax check timed out",
                syntax_ok=False,
            )
        except Exception as e:
            return SystemCommandOperation(
                success=False,
                command=cmd,
                reason=f"Syntax check error: {e}",
                syntax_ok=False,
            )

        if syntax.returncode != 0:
            return SystemCommandOperation(
                success=False,
                command=cmd,
                return_code=syntax.returncode,
                stdout=syntax.stdout,
                stderr=syntax.stderr,
                reason="Shell syntax validation failed",
                syntax_ok=False,
            )

        # 2) Execute
        try:
            run = subprocess.run(
                [self.shell, "-c", cmd],
                capture_output=True,
                text=True,
                cwd=self.cwd,
                env=self.env,
                timeout=self.timeout_seconds,
            )
            ok = run.returncode == 0
            return SystemCommandOperation(
                success=ok,
                command=cmd,
                return_code=run.returncode,
                stdout=run.stdout,
                stderr=run.stderr,
                reason=None if ok else "Command exited non-zero",
                syntax_ok=True,
            )
        except subprocess.TimeoutExpired as e:
            return SystemCommandOperation(
                success=False,
                command=cmd,
                return_code=None,
                stdout=getattr(e, "stdout", None),
                stderr=getattr(e, "stderr", None),
                reason="Command execution timed out",
                syntax_ok=True,
            )
        except Exception as e:
            return SystemCommandOperation(
                success=False,
                command=cmd,
                reason=f"Command execution error: {e}",
                syntax_ok=True,
            )
