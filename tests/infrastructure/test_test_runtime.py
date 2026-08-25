import os
import signal
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from fsrl.infrastructure.test_runtime import TIMEOUT_EXIT_CODE, run_command


@unittest.skipUnless(hasattr(os, "killpg") and Path("/proc").is_dir(), "requires Linux")
class TestRuntimeTests(unittest.TestCase):
    def _exercise_cleanup(self, *, interrupt: bool):
        program = (
            "import os, pathlib, signal, subprocess, sys, time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "child = subprocess.Popen([sys.executable, '-c', "
            "'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "time.sleep(60)']); "
            "pathlib.Path(sys.argv[1]).write_text("
            "f'{os.getpid()} {child.pid}', encoding='utf-8'); "
            "time.sleep(60)"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            pid_path = Path(temp_dir) / "pids.txt"
            pids = []
            timer = None
            try:
                if interrupt:
                    timer = threading.Timer(
                        1.0, os.kill, args=(os.getpid(), signal.SIGINT)
                    )
                    timer.start()
                started = time.monotonic()
                exit_code = run_command(
                    [sys.executable, "-c", program, str(pid_path)],
                    timeout_seconds=10.0 if interrupt else 1.0,
                    terminate_grace_seconds=0.1,
                )
                elapsed = time.monotonic() - started
                expected_exit_code = (
                    128 + signal.SIGINT if interrupt else TIMEOUT_EXIT_CODE
                )
                self.assertEqual(exit_code, expected_exit_code)
                self.assertLess(elapsed, 5.0)
                pids = [int(value) for value in pid_path.read_text().split()]
                deadline = time.monotonic() + 2.0
                while any(Path(f"/proc/{pid}").exists() for pid in pids):
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(0.01)
                self.assertTrue(all(not Path(f"/proc/{pid}").exists() for pid in pids))
            finally:
                if timer is not None:
                    timer.cancel()
                for pid in pids:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_timeout_kills_process_group_including_term_ignoring_child(self):
        self._exercise_cleanup(interrupt=False)

    def test_keyboard_interrupt_kills_process_group(self):
        self._exercise_cleanup(interrupt=True)
