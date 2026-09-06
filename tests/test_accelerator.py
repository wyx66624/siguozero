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


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
class PagedKVCudaTests(unittest.TestCase):
    def test_variable_length_decode_matches_individual_attention(self) -> None:
        from junqi.training.paged_kv import PagedKVCache

        device = torch.device("cuda")
        dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
        generator = torch.Generator(device=device).manual_seed(20260906)
        store = PagedKVCache(
            device=device,
            dtype=dtype,
            num_layers=1,
            num_heads=2,
            head_dim=4,
            max_tokens=32,
            max_entries=6,
        )
        lengths = (3, 16)
        source_keys = [
            torch.randn(
                (2, length, 4),
                device=device,
                dtype=dtype,
                generator=generator,
            )
            for length in lengths
        ]
        source_values = [
            torch.randn(
                (2, length, 4),
                device=device,
                dtype=dtype,
                generator=generator,
            )
            for length in lengths
        ]
        prefixes = [
            store.from_contiguous(
                [keys],
                [values],
                length=length,
                context=torch.zeros(8, device=device, dtype=dtype),
            )
            for keys, values, length in zip(
                source_keys, source_values, lengths, strict=True
            )
        ]
        children, pages, offsets = store.fork_for_append(prefixes)
        self.assertEqual(offsets.tolist(), [3, 0])

        new_keys = torch.randn(
            (2, 2, 4), device=device, dtype=dtype, generator=generator
        )
        new_values = torch.randn(
            (2, 2, 4), device=device, dtype=dtype, generator=generator
        )
        queries = torch.randn(
            (2, 2, 4), device=device, dtype=dtype, generator=generator
        )
        store.append(0, pages, offsets, new_keys, new_values)
        page_table, valid_tokens, maximum_length = store.make_page_table(children)
        self.assertIsNotNone(valid_tokens)
        assert valid_tokens is not None
        actual = store.decode(
            0, queries, page_table, valid_tokens, maximum_length
        )

        expected_rows = []
        for index, length in enumerate(lengths):
            keys = torch.cat(
                (source_keys[index], new_keys[index].unsqueeze(1)), dim=1
            )
            values = torch.cat(
                (source_values[index], new_values[index].unsqueeze(1)), dim=1
            )
            expected_rows.append(
                torch.nn.functional.scaled_dot_product_attention(
                    queries[index].unsqueeze(0).unsqueeze(2),
                    keys.unsqueeze(0),
                    values.unsqueeze(0),
                    dropout_p=0.0,
                    is_causal=False,
                ).squeeze(0).squeeze(1)
            )
        expected = torch.stack(expected_rows)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        self.assertEqual(valid_tokens[:, 0, 0].sum(dim=1).tolist(), [4, 17])

        for state in children:
            store.release(state)
        for state in prefixes:
            store.release(state)
        self.assertEqual(store.used_pages, 0)


if __name__ == "__main__":
    unittest.main()
