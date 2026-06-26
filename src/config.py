"""Configuration management for Deterministic Horizon experiments."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from omegaconf import OmegaConf


@dataclass
class ModelConfig:
    """Model configuration."""

    name: str = ""
    provider: str = ""
    temperature: float = 0.0
    max_tokens: int = 8192  # paper setting
    timeout: int = 120
    max_retries: int = 3

    # Local model specific
    model_path: str | None = None
    device: str = "auto"
    load_in_8bit: bool = False
    load_in_4bit: bool = False


@dataclass
class TaskConfig:
    """Task configuration."""

    name: str = "permutation"
    n_elements: int = 8
    # Canonical PermutationProbe operators: adjacent transpositions on S_n.
    operators: list[str] = field(default_factory=lambda: [f"swap_{i}{i + 1}" for i in range(7)])

    # Instance generation (max depth is the diameter C(n,2) = 28 for n=8).
    min_depth: int = 4
    max_depth: int = 28
    depth_step: int = 4
    instances_per_depth: int = 100


@dataclass
class ExperimentConfig:
    """Experiment configuration."""

    name: str = "default"
    seeds: list[int] = field(default_factory=lambda: [42, 2024, 2025])
    n_instances: int = 500
    conditions: list[str] = field(default_factory=lambda: ["C1", "C2", "C3"])

    # Evaluation
    batch_size: int = 50
    save_traces: bool = True
    compute_ssj: bool = True
    compute_sfe: bool = True


@dataclass
class OutputConfig:
    """Output configuration."""

    dir: str = "results"
    save_traces: bool = True
    save_metrics: bool = True
    generate_figures: bool = True
    figure_format: str = "pdf"


@dataclass
class Config:
    """Main configuration container."""

    model: ModelConfig = field(default_factory=ModelConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Determinism
    random_seed: int = 42
    deterministic: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.experiment.n_instances < 1:
            raise ValueError("n_instances must be positive")
        if not self.experiment.seeds:
            raise ValueError("At least one seed must be specified")
        valid_conditions = {"C1", "C2", "C3", "C4", "C5"}
        invalid = set(self.experiment.conditions) - valid_conditions
        if invalid:
            raise ValueError(f"Invalid conditions: {invalid}")


def load_config(
    config_path: str | Path | None = None, overrides: list[str] | None = None
) -> Config:
    """
    Load configuration from YAML file with optional overrides.

    Args:
        config_path: Path to YAML config file
        overrides: List of override strings (e.g., ["model.name=gpt-4o"])

    Returns:
        Validated Config object
    """
    # Start with defaults
    base_config = OmegaConf.structured(Config)

    # Load from file if provided
    if config_path is not None:
        config_path = Path(config_path)
        if config_path.exists():
            file_config = OmegaConf.load(config_path)
            base_config = OmegaConf.merge(base_config, file_config)

    # Apply command-line overrides
    if overrides:
        override_config = OmegaConf.from_dotlist(overrides)
        base_config = OmegaConf.merge(base_config, override_config)

    # Convert to Config dataclass
    config_dict = OmegaConf.to_container(base_config, resolve=True)

    return Config(
        model=ModelConfig(**config_dict.get("model", {})),
        task=TaskConfig(**config_dict.get("task", {})),
        experiment=ExperimentConfig(**config_dict.get("experiment", {})),
        output=OutputConfig(**config_dict.get("output", {})),
        random_seed=config_dict.get("random_seed", 42),
        deterministic=config_dict.get("deterministic", True),
    )


def save_config(config: Config, path: str | Path) -> None:
    """Save configuration to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = {
        "model": {
            "name": config.model.name,
            "provider": config.model.provider,
            "temperature": config.model.temperature,
            "max_tokens": config.model.max_tokens,
            "timeout": config.model.timeout,
            "max_retries": config.model.max_retries,
        },
        "task": {
            "name": config.task.name,
            "n_elements": config.task.n_elements,
            "operators": config.task.operators,
            "min_depth": config.task.min_depth,
            "max_depth": config.task.max_depth,
            "depth_step": config.task.depth_step,
            "instances_per_depth": config.task.instances_per_depth,
        },
        "experiment": {
            "name": config.experiment.name,
            "seeds": config.experiment.seeds,
            "n_instances": config.experiment.n_instances,
            "conditions": config.experiment.conditions,
            "batch_size": config.experiment.batch_size,
            "save_traces": config.experiment.save_traces,
        },
        "output": {
            "dir": config.output.dir,
            "save_traces": config.output.save_traces,
            "save_metrics": config.output.save_metrics,
            "generate_figures": config.output.generate_figures,
            "figure_format": config.output.figure_format,
        },
        "random_seed": config.random_seed,
        "deterministic": config.deterministic,
    }

    with open(path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
