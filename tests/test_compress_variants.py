"""Tests for compress_variants.py magnitude pruning CLI."""

import tempfile
import torch
from pathlib import Path


def test_smoke_pruning_achieves_target_sparsity():
    """End-to-end smoke test on tiny-gpt2: asserts sparsity within 0.01 of target."""
    import subprocess

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "tiny_mag10"
        result = subprocess.run(
            [
                "/opt/miniconda3/envs/jrp/bin/python",
                "projects/02-compression-tampering/compress_variants.py",
                "--sparsity", "0.10",
                "--model", "sshleifer/tiny-gpt2",  # Tiny model for fast test
                "--out", str(out_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        # Load model and verify sparsity
        from transformers import AutoModelForCausalLM
        import torch.nn as nn

        model = AutoModelForCausalLM.from_pretrained(out_path, local_files_only=True)

        total_params = 0
        zero_params = 0
        for name, module in model.named_modules():
            # Handle both Linear and Conv1D (GPT-2 uses Conv1D)
            is_linear = isinstance(module, nn.Linear)
            is_conv1d = type(module).__name__ == "Conv1D"
            if not (is_linear or is_conv1d):
                continue
            # Exclude embeddings and lm_head per plan
            if "embed" in name or "lm_head" in name:
                continue
            total_params += module.weight.numel()
            zero_params += (module.weight == 0.0).sum().item()

        achieved_sparsity = zero_params / total_params
        target = 0.10

        assert (
            abs(achieved_sparsity - target) < 0.01
        ), f"Sparsity {achieved_sparsity:.4f} not within 0.01 of target {target}"


def test_embeddings_and_lm_head_excluded():
    """Verify embeddings and lm_head are never pruned."""
    import subprocess

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "tiny_mag50"
        result = subprocess.run(
            [
                "/opt/miniconda3/envs/jrp/bin/python",
                "projects/02-compression-tampering/compress_variants.py",
                "--sparsity", "0.50",
                "--model", "sshleifer/tiny-gpt2",
                "--out", str(out_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        from transformers import AutoModelForCausalLM
        import torch.nn as nn

        model = AutoModelForCausalLM.from_pretrained(out_path, local_files_only=True)

        # Check embeddings have no zeros (check all parameters)
        for name, param in model.named_parameters():
            if "embed" in name:
                assert (param != 0.0).all(), f"{name} should have no zeros"

        # Check lm_head has no zeros (may be Linear or Conv1D)
        is_linear = isinstance(model.lm_head, nn.Linear)
        is_conv1d = type(model.lm_head).__name__ == "Conv1D"
        if is_linear or is_conv1d:
            assert (model.lm_head.weight != 0.0).all(), "lm_head weight should have no zeros"
        else:
            # Fallback to parameter check
            for name, param in model.named_parameters():
                if "lm_head" in name:
                    assert (param != 0.0).all(), f"{name} should have no zeros"
