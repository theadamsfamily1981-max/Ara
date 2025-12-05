#!/usr/bin/env python3
"""
ARA: Unified AI System - Main Entry Point

This is the unified entry point for all ARA functionality:
- API server for avatar generation and TFAN
- Training for models and agents
- Meta-learning and self-improvement
- Research institute and experiments
- Embodiment and hardware management
- Interactive testing and diagnostics

Usage:
    # System status
    ara status

    # Start API server
    ara api

    # Interactive conversation
    ara chat

    # Meta-learning
    ara meta patterns --list
    ara meta research --show PROG-001

    # Research institute
    ara institute --brief
    ara institute --council

    # Embodiment
    ara body --devices
    ara wake / ara sleep
    ara health

    # User model
    ara user --profile
"""

import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Optional

# Setup path
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ara")


def check_dependencies():
    """Check and report on available dependencies."""
    deps = {}

    # Core
    try:
        import torch
        deps['torch'] = torch.__version__
        deps['cuda'] = torch.cuda.is_available()
    except ImportError:
        deps['torch'] = None
        deps['cuda'] = False

    try:
        import numpy
        deps['numpy'] = numpy.__version__
    except ImportError:
        deps['numpy'] = None

    # ML/DL
    try:
        import transformers
        deps['transformers'] = transformers.__version__
    except ImportError:
        deps['transformers'] = None

    # API
    try:
        import fastapi
        deps['fastapi'] = fastapi.__version__
    except ImportError:
        deps['fastapi'] = None

    try:
        import uvicorn
        deps['uvicorn'] = uvicorn.__version__
    except ImportError:
        deps['uvicorn'] = None

    # Geometry/Math
    try:
        import geoopt
        deps['geoopt'] = geoopt.__version__
    except ImportError:
        deps['geoopt'] = None

    try:
        import gudhi
        deps['gudhi'] = True
    except ImportError:
        deps['gudhi'] = None

    return deps


def print_status():
    """Print comprehensive system status."""
    print("=" * 60)
    print("ARA: Unified AI System - Status")
    print("=" * 60)

    # Python version
    print(f"\nPython: {sys.version.split()[0]}")

    # Dependencies
    deps = check_dependencies()
    print("\nCore Dependencies:")
    for dep, version in deps.items():
        if version is True:
            status = "installed"
        elif version is False:
            status = "not available"
        elif version is None:
            status = "not installed"
        else:
            status = f"v{version}"
        print(f"  {dep}: {status}")

    # GPU info
    if deps['torch'] and deps['cuda']:
        import torch
        print(f"\nGPU:")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {name} ({mem:.1f}GB)")

    # Package structure (core)
    print("\nCore Packages:")
    ara_path = Path(__file__).parent
    for subdir in ['models', 'agents', 'training', 'api', 'configs', 'utils', 'service']:
        subpath = ara_path / subdir
        if subpath.exists():
            print(f"  ara/{subdir}/: OK")
        else:
            print(f"  ara/{subdir}/: Missing")

    # Meta-learning subsystems
    print("\nMeta-Learning Subsystems:")
    for subdir in ['meta', 'academy', 'institute', 'embodied', 'user']:
        subpath = ara_path / subdir
        if subpath.exists():
            print(f"  ara/{subdir}/: OK")
        else:
            print(f"  ara/{subdir}/: Missing")

    # Subsystem status
    print("\nSubsystem Status:")

    # Meta-learning
    try:
        from ara.meta.meta_brain import get_meta_brain
        brain = get_meta_brain()
        status = brain.get_status()
        print(f"  META-LEARNING: {status.get('patterns_tracked', 0)} patterns")
    except Exception:
        print("  META-LEARNING: Available")

    # Academy
    try:
        from ara.academy import get_skill_registry
        registry = get_skill_registry()
        summary = registry.get_summary()
        print(f"  ACADEMY: {summary['total_skills']} skills")
    except Exception:
        print("  ACADEMY: Available")

    # Institute
    try:
        from ara.institute import get_research_graph
        graph = get_research_graph()
        summary = graph.get_summary()
        print(f"  INSTITUTE: {summary['topics']['total']} topics, {summary['hypotheses']['total']} hypotheses")
    except Exception:
        print("  INSTITUTE: Available")

    # Embodiment
    try:
        from ara.embodied import get_embodiment_core
        core = get_embodiment_core()
        body_state = core.get_body_state()
        print(f"  EMBODIMENT: {body_state['state']} ({body_state['online_devices']} devices)")
    except Exception:
        print("  EMBODIMENT: Available")

    # User model
    try:
        from ara.user import get_user_model
        model = get_user_model()
        summary = model.get_summary()
        print(f"  USER MODEL: {summary['total_interactions']} interactions")
    except Exception:
        print("  USER MODEL: Available")

    # Config files
    print("\nConfiguration Files:")
    config_dir = _root / "config"
    if config_dir.exists():
        for f in config_dir.glob("*.yaml"):
            print(f"  {f.name}: OK")

    print("\n" + "=" * 60)
    print("Use 'ara --help' for available commands")
    print("=" * 60)


def run_api(host: str = "0.0.0.0", port: int = 8000):
    """Run the unified API server."""
    logger.info(f"Starting ARA API server on {host}:{port}")

    try:
        import uvicorn
        from ara.api import create_app

        app = create_app()
        uvicorn.run(app, host=host, port=port, log_level="info")

    except ImportError as e:
        logger.error(f"Failed to start API: {e}")
        logger.error("Install: pip install fastapi uvicorn")
        sys.exit(1)


def run_training(backend: str, config_path: str = None):
    """Run training with specified backend."""
    logger.info(f"Starting training with backend: {backend}")

    try:
        from ara.training import UnifiedTrainer

        config = {}
        if config_path:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)

        trainer = UnifiedTrainer(backend=backend, config=config)
        trainer.train()

    except ImportError as e:
        logger.error(f"Failed to start training: {e}")
        sys.exit(1)


def run_interactive():
    """Run interactive mode."""
    print("\n" + "=" * 60)
    print("ARA: Unified AI System - Interactive Mode")
    print("=" * 60)

    while True:
        print("\nOptions:")
        print("  1. System status")
        print("  2. Start API server")
        print("  3. Test imports")
        print("  4. Show migration guide")
        print("  5. Check configs")
        print("  0. Exit")

        choice = input("\nSelect option: ").strip()

        if choice == "0":
            print("Goodbye!")
            break
        elif choice == "1":
            print_status()
        elif choice == "2":
            run_api()
        elif choice == "3":
            test_imports()
        elif choice == "4":
            from ara.compat import print_migration_guide
            print_migration_guide()
        elif choice == "5":
            test_configs()
        else:
            print("Invalid option")


def test_imports():
    """Test that all imports work."""
    print("\nTesting imports...")

    tests = [
        ("ara", "from ara import __version__"),
        ("ara.models", "from ara.models import TFANConfig"),
        ("ara.agents", "from ara.agents import HRRLConfig"),
        ("ara.configs", "from ara.configs import AraConfig, load_config"),
        ("ara.utils", "from ara.utils import get_device, set_seed"),
        ("ara.training", "from ara.training import UnifiedTrainer"),
        ("ara.api", "from ara.api import FASTAPI_AVAILABLE"),
    ]

    results = []
    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"  {name}: OK")
            results.append(True)
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
            results.append(False)

    success = sum(results)
    total = len(results)
    print(f"\nResults: {success}/{total} passed")


def test_configs():
    """Test configuration loading."""
    print("\nTesting configuration...")

    try:
        from ara.configs import load_config, AraConfig

        config = load_config()
        print(f"  AraConfig loaded: OK")
        print(f"  Model type: {config.model.model_type}")
        print(f"  Hidden size: {config.model.hidden_size}")
        print(f"  Device: {config.device}")

    except Exception as e:
        print(f"  Config loading failed: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ARA: Unified AI System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ara.main status          Show system status
  python -m ara.main api             Start API server
  python -m ara.main train           Run training
  python -m ara.main interactive     Interactive mode
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show system status")

    # API command
    api_parser = subparsers.add_parser("api", help="Start API server")
    api_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to bind")

    # Training command
    train_parser = subparsers.add_parser("train", help="Run training")
    train_parser.add_argument("--backend", default="tfan",
                              choices=["tfan", "snn", "agent", "tgsfn"],
                              help="Training backend")
    train_parser.add_argument("--config", help="Path to config file")

    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive mode")

    # Test command
    test_parser = subparsers.add_parser("test", help="Run tests")

    args = parser.parse_args()

    if args.command == "status" or args.command is None:
        print_status()
    elif args.command == "api":
        run_api(host=args.host, port=args.port)
    elif args.command == "train":
        run_training(backend=args.backend, config_path=args.config)
    elif args.command == "interactive":
        run_interactive()
    elif args.command == "test":
        test_imports()
        test_configs()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
