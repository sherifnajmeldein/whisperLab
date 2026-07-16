import threading
import time
import unittest
from unittest.mock import patch

import app


class ModelLoadingTests(unittest.TestCase):
    def test_concurrent_first_requests_load_one_model(self) -> None:
        """Only one Whisper model is constructed when requests arrive together."""
        calls = 0
        calls_lock = threading.Lock()

        class FakeModel:
            def __init__(self, *args: object, **kwargs: object) -> None:
                nonlocal calls
                with calls_lock:
                    calls += 1
                time.sleep(0.05)

        start = threading.Barrier(2)
        models: list[FakeModel] = []

        def load_model() -> None:
            start.wait()
            model, _ = app.get_model()
            models.append(model)

        with patch.object(app, "WhisperModel", FakeModel), patch.object(app, "_model", None):
            threads = [threading.Thread(target=load_model) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        self.assertEqual(calls, 1)
        self.assertEqual(len(models), 2)
        self.assertIs(models[0], models[1])
