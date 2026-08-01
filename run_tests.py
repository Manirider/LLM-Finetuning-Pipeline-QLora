#!/usr/bin/env python
"""
Test runner script for CI/CD pipeline.

Runs all tests with coverage reporting.
"""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run all tests with coverage."""
    
    # Change to project root
    project_root = Path(__file__).parent
    
    # Run pytest with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "--cov-report=xml:coverage.xml",
        "--cov-fail-under=90",
        "-x",  # Stop on first failure
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print(f"Working directory: {project_root}")
    
    result = subprocess.run(cmd, cwd=project_root)
    
    return result.returncode


def run_unit_tests():
    """Run only unit tests."""
    project_root = Path(__file__).parent
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/unit/",
        "-v",
        "--tb=short",
        "--cov=src",
        "--cov-report=term-missing",
    ]
    
    print(f"Running unit tests: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode


def run_integration_tests():
    """Run only integration tests."""
    project_root = Path(__file__).parent
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/integration/",
        "-v",
        "--tb=short",
        "--cov=src",
        "--cov-report=term-missing",
    ]
    
    print(f"Running integration tests: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode


def run_smoke_tests():
    """Run smoke tests."""
    project_root = Path(__file__).parent
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/smoke/",
        "-v",
        "--tb=short",
    ]
    
    print(f"Running smoke tests: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode


def run_pipeline_tests():
    """Run pipeline tests."""
    project_root = Path(__file__).parent
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/pipeline/",
        "-v",
        "--tb=short",
        "--cov=src",
        "--cov-report=term-missing",
    ]
    
    print(f"Running pipeline tests: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode


def run_lint():
    """Run linting checks."""
    project_root = Path(__file__).parent
    
    cmd = [
        sys.executable, "-m", "ruff",
        "check", "src/", "tests/",
    ]
    
    print(f"Running lint: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode


def run_format_check():
    """Run format check."""
    project_root = Path(__file__).parent
    
    cmd = [
        sys.executable, "-m", "ruff",
        "format", "--check", "src/", "tests/",
    ]
    
    print(f"Running format check: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode


def run_type_check():
    """Run type checking."""
    project_root = Path(__file__).parent
    
    cmd = [
        sys.executable, "-m", "mypy",
        "src/",
    ]
    
    print(f"Running type check: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test runner for LLM Fine-tuning Pipeline")
    parser.add_argument(
        "--suite",
        choices=["all", "unit", "integration", "smoke", "pipeline", "lint", "format", "type"],
        default="all",
        help="Test suite to run",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failure",
    )
    
    args = parser.parse_args()
    
    if args.suite == "all":
        exit_code = run_tests()
    elif args.suite == "unit":
        exit_code = run_unit_tests()
    elif args.suite == "integration":
        exit_code = run_integration_tests()
    elif args.suite == "smoke":
        exit_code = run_smoke_tests()
    elif args.suite == "pipeline":
        exit_code = run_pipeline_tests()
    elif args.suite == "lint":
        exit_code = run_lint()
    elif args.suite == "format":
        exit_code = run_format_check()
    elif args.suite == "type":
        exit_code = run_type_check()
    else:
        print(f"Unknown suite: {args.suite}")
        exit_code = 1
    
    if exit_code != 0:
        print(f"\nTests failed with exit code {exit_code}")
        sys.exit(exit_code)
    
    print("\nAll tests passed!")
    sys.exit(0)


if __name__ == "__main__":
    main()