"""Timeout-bounded test entry point with process-group cleanup."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_TERMINATE_GRACE_SECONDS = 2.0
TIMEOUT_EXIT_CODE = 124


class _ForwardedSignal(Exception):
    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_for_process_group_exit(
    process: subprocess.Popen, process_group_id: int, timeout_seconds: float
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        process.poll()
        if not _process_group_exists(process_group_id):
            return True
        time.sleep(0.01)
    process.poll()
    return not _process_group_exists(process_group_id)


def _terminate_process_group(
    process: subprocess.Popen, terminate_grace_seconds: float
) -> None:
    process_group_id = process.pid
    if not _process_group_exists(process_group_id):
        process.wait()
        return

    os.killpg(process_group_id, signal.SIGTERM)
    if not _wait_for_process_group_exit(
        process, process_group_id, terminate_grace_seconds
    ):
        os.killpg(process_group_id, signal.SIGKILL)
        if not _wait_for_process_group_exit(
            process, process_group_id, max(1.0, terminate_grace_seconds)
        ):
            raise RuntimeError(
                f"test process group {process_group_id} survived SIGKILL"
            )
    process.wait()


def _raise_forwarded_signal(signum, _frame) -> None:
    raise _ForwardedSignal(signum)


def run_command(
    command: list[str], *, timeout_seconds: float, terminate_grace_seconds: float
) -> int:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if terminate_grace_seconds < 0:
        raise ValueError("terminate_grace_seconds must be nonnegative")

    process = subprocess.Popen(command, start_new_session=True)
    handled_signals = tuple(
        candidate
        for candidate in (signal.SIGTERM, getattr(signal, "SIGHUP", None))
        if candidate is not None
    )
    previous_handlers = {
        candidate: signal.signal(candidate, _raise_forwarded_signal)
        for candidate in handled_signals
    }
    try:
        return process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        print(
            f"Test command timed out after {timeout_seconds:g}s; "
            f"terminating process group {process.pid}.",
            file=sys.stderr,
        )
        _terminate_process_group(process, terminate_grace_seconds)
        return TIMEOUT_EXIT_CODE
    except KeyboardInterrupt:
        print(
            f"Test command interrupted; terminating process group {process.pid}.",
            file=sys.stderr,
        )
        _terminate_process_group(process, terminate_grace_seconds)
        return 128 + signal.SIGINT
    except _ForwardedSignal as interrupted:
        print(
            f"Test runtime received signal {interrupted.signum}; "
            f"terminating process group {process.pid}.",
            file=sys.stderr,
        )
        _terminate_process_group(process, terminate_grace_seconds)
        return 128 + interrupted.signum
    finally:
        for candidate, previous_handler in previous_handlers.items():
            signal.signal(candidate, previous_handler)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Run unit tests with a timeout and whole-process-group cleanup."
    )
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, metavar="SECONDS"
    )
    parser.add_argument(
        "--terminate-grace",
        type=float,
        default=DEFAULT_TERMINATE_GRACE_SECONDS,
        metavar="SECONDS",
    )
    parser.add_argument("--framework", choices=("pytest", "unittest"), default="pytest")
    parser.add_argument("test_args", nargs=argparse.REMAINDER)
    return parser.parse_args(args)


def main(args=None) -> int:
    parsed = parse_args(args)
    test_args = list(parsed.test_args)
    if test_args[:1] == ["--"]:
        test_args.pop(0)
    if not test_args and parsed.framework == "pytest":
        test_args = ["-q"]
    command = [sys.executable, "-m", parsed.framework, *test_args]
    return run_command(
        command,
        timeout_seconds=parsed.timeout,
        terminate_grace_seconds=parsed.terminate_grace,
    )


if __name__ == "__main__":
    raise SystemExit(main())
