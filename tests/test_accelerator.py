from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from junqi.training import accelerator


class AcceleratorCompatibilityTests(unittest.TestCase):
    def test_cpu_resolution_does_not_require_torch_npu(self) -> None:
        self.assertEqual(accelerator.resolve_device("cpu"), torch.device("cpu"))

    def test_npu_runtime_accepts_the_pinned_27_stack(self) -> None:
        torch_npu = SimpleNamespace(__version__="2.7.1.post8")
        with (
            patch.object(accelerator, "load_torch_npu", return_value=torch_npu),
            patch.object(torch, "__version__", "2.7.1+cpu"),
        ):
            accelerator.validate_npu_runtime()

    def test_npu_runtime_rejects_a_different_torch_patch(self) -> None:
        torch_npu = SimpleNamespace(__version__="2.7.1.post8")
        with (
            patch.object(accelerator, "load_torch_npu", return_value=torch_npu),
            patch.object(torch, "__version__", "2.7.0"),
            self.assertRaisesRegex(RuntimeError, "PyTorch 2.7.1 exactly"),
        ):
            accelerator.validate_npu_runtime()

    def test_npu_runtime_rejects_numpy_2(self) -> None:
        torch_npu = SimpleNamespace(__version__="2.7.1.post8")
        numpy = SimpleNamespace(__version__="2.0.0")
        with (
            patch.object(accelerator, "load_torch_npu", return_value=torch_npu),
            patch.object(torch, "__version__", "2.7.1"),
            patch.object(accelerator.importlib, "import_module", return_value=numpy),
            self.assertRaisesRegex(RuntimeError, "NumPy <2"),
        ):
            accelerator.validate_npu_runtime()

    def test_npu_oom_runtime_error_is_detected(self) -> None:
        self.assertTrue(
            accelerator.is_out_of_memory(
                RuntimeError("NPU out of memory while allocating"), "npu"
            )
        )


if __name__ == "__main__":
    unittest.main()
