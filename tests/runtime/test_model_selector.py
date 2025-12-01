"""
Tests for runtime model selector.

Validates that ModelSelector correctly loads configs, applies overrides,
and handles missing files gracefully.
"""

import pytest
import yaml
import json
from pathlib import Path
from tfan.runtime import ModelSelector, ModelConfig
from tfan.runtime.model_selector import create_config_from_pareto_result


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_config(self):
        """Test default ModelConfig creation."""
        config = ModelConfig()
        assert config.n_heads == 8
        assert config.d_model == 512
        assert config.n_layers == 12
        assert config.keep_ratio == 1.0
        assert config.source == "default"

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "n_heads": 16,
            "d_model": 1024,
            "keep_ratio": 0.7,
            "latency_ms": 123.4,
            "unknown_field": "should be ignored",
        }
        config = ModelConfig.from_dict(data)
        assert config.n_heads == 16
        assert config.d_model == 1024
        assert config.keep_ratio == 0.7
        assert config.latency_ms == 123.4
        # Unknown fields should be ignored, not cause errors
        assert not hasattr(config, "unknown_field")

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = ModelConfig(n_heads=16, d_model=1024, latency_ms=100.0)
        data = config.to_dict()
        assert data["n_heads"] == 16
        assert data["d_model"] == 1024
        assert data["latency_ms"] == 100.0

    def test_round_trip(self):
        """Test dict -> config -> dict round trip."""
        original = {
            "n_heads": 12,
            "d_model": 768,
            "n_layers": 16,
            "keep_ratio": 0.85,
        }
        config = ModelConfig.from_dict(original)
        recovered = config.to_dict()
        for key in original:
            assert recovered[key] == original[key]


class TestModelSelector:
    """Tests for ModelSelector class."""

    @pytest.fixture
    def temp_config_file(self, tmp_path):
        """Create a temporary config file."""
        config_dir = tmp_path / "configs" / "auto"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "best.yaml"

        config_data = {
            "n_heads": 16,
            "d_model": 1024,
            "n_layers": 24,
            "keep_ratio": 0.75,
            "latency_ms": 150.0,
            "accuracy": 0.92,
        }

        with open(config_path, "w") as f:
            yaml.safe_dump(config_data, f)

        return config_path

    def test_load_config_from_file(self, temp_config_file):
        """Test loading config from YAML file."""
        selector = ModelSelector(config_path=temp_config_file)
        config = selector.get_config()

        assert config.n_heads == 16
        assert config.d_model == 1024
        assert config.n_layers == 24
        assert config.keep_ratio == 0.75
        assert config.latency_ms == 150.0
        assert config.accuracy == 0.92
        assert str(temp_config_file) in config.source

    def test_config_caching(self, temp_config_file):
        """Test that config is cached after first load."""
        selector = ModelSelector(config_path=temp_config_file)

        config1 = selector.get_config()
        config2 = selector.get_config()

        assert config1 is config2  # Same object

    def test_reload(self, temp_config_file):
        """Test reloading config from disk."""
        selector = ModelSelector(config_path=temp_config_file)

        config1 = selector.get_config()
        assert config1.n_heads == 16

        # Modify the file
        with open(temp_config_file, "r") as f:
            data = yaml.safe_load(f)
        data["n_heads"] = 32
        with open(temp_config_file, "w") as f:
            yaml.safe_dump(data, f)

        # Reload
        config2 = selector.reload()
        assert config2.n_heads == 32
        assert config2 is not config1  # Different object

    def test_overrides(self, temp_config_file):
        """Test applying parameter overrides."""
        overrides = {"n_heads": 32, "d_model": 2048}
        selector = ModelSelector(config_path=temp_config_file, overrides=overrides)
        config = selector.get_config()

        # Overridden values
        assert config.n_heads == 32
        assert config.d_model == 2048

        # Original values preserved
        assert config.n_layers == 24
        assert config.keep_ratio == 0.75

    def test_missing_file_with_fallback(self, tmp_path):
        """Test behavior when config file is missing (non-strict)."""
        missing_path = tmp_path / "nonexistent.yaml"
        selector = ModelSelector(config_path=missing_path, strict=False)
        config = selector.get_config()

        # Should get default config
        assert config.n_heads == 8
        assert config.d_model == 512
        assert config.source in ("default", str(selector.FALLBACK_CONFIG_PATH))

    def test_missing_file_strict(self, tmp_path):
        """Test that strict mode raises error for missing file."""
        missing_path = tmp_path / "nonexistent.yaml"
        selector = ModelSelector(config_path=missing_path, strict=True)

        with pytest.raises(FileNotFoundError):
            selector.get_config()

    def test_from_cli_override(self):
        """Test creating selector from CLI JSON string."""
        override_str = '{"n_heads": 24, "d_model": 1536}'
        selector = ModelSelector.from_cli_override(override_str)

        assert selector.overrides == {"n_heads": 24, "d_model": 1536}

    def test_from_cli_override_invalid_json(self):
        """Test that invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            ModelSelector.from_cli_override("{invalid json}")

    def test_summary(self, temp_config_file):
        """Test summary string generation."""
        selector = ModelSelector(config_path=temp_config_file)
        summary = selector.summary()

        assert "24L" in summary  # n_layers
        assert "16H" in summary  # n_heads
        assert "1024D" in summary  # d_model
        assert "150.0ms" in summary  # latency
        assert "0.920" in summary  # accuracy

    def test_summary_with_overrides(self, temp_config_file):
        """Test summary includes override information."""
        overrides = {"n_heads": 32}
        selector = ModelSelector(config_path=temp_config_file, overrides=overrides)
        summary = selector.summary()

        assert "Overrides:" in summary
        assert "n_heads" in summary


class TestParetoIntegration:
    """Tests for integration with Pareto optimization results."""

    def test_create_config_from_pareto_result(self):
        """Test creating ModelConfig from Pareto optimization output."""
        pareto_config = {
            "n_heads": 16,
            "d_model": 1024,
            "n_layers": 20,
            "keep_ratio": 0.8,
        }

        objectives = {
            "neg_accuracy": 0.08,  # 92% accuracy
            "latency": 140.0,
            "epr_cv": 0.12,
            "topo_gap": 0.015,
            "energy": 8.5,
        }

        config = create_config_from_pareto_result(
            pareto_config=pareto_config,
            objectives=objectives,
            rank=0,
            timestamp="2025-01-15T10:30:00",
        )

        # Architecture params
        assert config.n_heads == 16
        assert config.d_model == 1024
        assert config.keep_ratio == 0.8

        # Performance metrics
        assert config.accuracy == pytest.approx(0.92)
        assert config.latency_ms == 140.0
        assert config.epr_cv == 0.12
        assert config.topo_gap == 0.015
        assert config.energy_j == 8.5

        # Metadata
        assert config.pareto_rank == 0
        assert config.timestamp == "2025-01-15T10:30:00"
        assert config.source == "pareto_optimization"

    def test_export_and_load_pareto_config(self, tmp_path):
        """Test full cycle: Pareto result -> export -> load."""
        # Create config from Pareto result
        pareto_config = {"n_heads": 12, "d_model": 768, "keep_ratio": 0.9}
        objectives = {"neg_accuracy": 0.05, "latency": 100.0, "epr_cv": 0.10}

        config = create_config_from_pareto_result(
            pareto_config=pareto_config,
            objectives=objectives,
            rank=1,
        )

        # Export to YAML
        export_path = tmp_path / "exported_config.yaml"
        with open(export_path, "w") as f:
            yaml.safe_dump(config.to_dict(), f)

        # Load via ModelSelector
        selector = ModelSelector(config_path=export_path)
        loaded_config = selector.get_config()

        # Verify round-trip
        assert loaded_config.n_heads == 12
        assert loaded_config.d_model == 768
        assert loaded_config.keep_ratio == 0.9
        assert loaded_config.accuracy == pytest.approx(0.95)
        assert loaded_config.latency_ms == 100.0
        assert loaded_config.pareto_rank == 1


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_unknown_override_key(self, tmp_path, caplog):
        """Test that unknown override keys log a warning."""
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump({"n_heads": 8}, f)

        overrides = {"n_heads": 16, "unknown_param": 999}
        selector = ModelSelector(config_path=config_path, overrides=overrides)

        with caplog.at_level("WARNING"):
            config = selector.get_config()

        assert "Unknown override key" in caplog.text
        assert config.n_heads == 16  # Valid override applied

    def test_empty_config_file(self, tmp_path):
        """Test handling of empty config file."""
        config_path = tmp_path / "empty.yaml"
        config_path.write_text("")

        selector = ModelSelector(config_path=config_path)
        config = selector.get_config()

        # Should use defaults for missing fields
        assert config.n_heads == 8
        assert config.d_model == 512

    def test_partial_config_file(self, tmp_path):
        """Test handling of partial config (only some fields)."""
        config_path = tmp_path / "partial.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump({"n_heads": 24}, f)

        selector = ModelSelector(config_path=config_path)
        config = selector.get_config()

        # Specified field
        assert config.n_heads == 24

        # Default fields
        assert config.d_model == 512
        assert config.n_layers == 12


class TestCLIIntegration:
    """Test CLI integration scenarios."""

    def test_cli_override_workflow(self, tmp_path):
        """Simulate CLI usage with --config-override flag."""
        # Create base config
        config_path = tmp_path / "base.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump({"n_heads": 8, "d_model": 512}, f)

        # Simulate CLI override
        override_str = '{"n_heads": 16}'
        selector = ModelSelector.from_cli_override(override_str)
        selector.config_path = config_path

        config = selector.get_config()

        assert config.n_heads == 16  # Overridden
        assert config.d_model == 512  # From file

    def test_multiple_override_sources(self, tmp_path):
        """Test that direct overrides take precedence."""
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump({"n_heads": 8, "d_model": 512}, f)

        # Both constructor and file
        selector = ModelSelector(
            config_path=config_path, overrides={"n_heads": 32, "d_model": 2048}
        )

        config = selector.get_config()

        # Overrides should win
        assert config.n_heads == 32
        assert config.d_model == 2048


# Hard gate tests
class TestHardGates:
    """Verify hard gates for model selector functionality."""

    def test_gate_config_load_latency(self, temp_config_file, benchmark):
        """Gate: Config load latency < 10ms."""

        def load_config():
            selector = ModelSelector(config_path=temp_config_file)
            return selector.get_config()

        result = benchmark(load_config)
        assert result.n_heads == 16

        # Gate: latency < 10ms (generous for file I/O)
        assert benchmark.stats["mean"] < 0.010

    def test_gate_override_application(self, temp_config_file):
        """Gate: Overrides must be correctly applied to all fields."""
        overrides = {
            "n_heads": 32,
            "d_model": 2048,
            "keep_ratio": 0.5,
            "dph_enabled": True,
        }

        selector = ModelSelector(config_path=temp_config_file, overrides=overrides)
        config = selector.get_config()

        # Gate: All overrides must be applied
        for key, expected_value in overrides.items():
            actual_value = getattr(config, key)
            assert (
                actual_value == expected_value
            ), f"Override {key} not applied: {actual_value} != {expected_value}"

    def test_gate_missing_file_graceful_degradation(self, tmp_path):
        """Gate: Missing config file must not crash (non-strict mode)."""
        missing_path = tmp_path / "missing.yaml"

        # Should not raise exception
        selector = ModelSelector(config_path=missing_path, strict=False)
        config = selector.get_config()

        # Should return valid config
        assert isinstance(config, ModelConfig)
        assert config.n_heads > 0
        assert config.d_model > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
