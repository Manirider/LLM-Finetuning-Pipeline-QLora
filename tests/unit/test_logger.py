"""Unit tests for src/logger.py."""

import json
import logging

from src.logger import (
    ColoredConsoleFormatter,
    StructuredFormatter,
    get_logger,
    setup_logging,
)


class TestStructuredFormatter:
    """Tests for StructuredFormatter."""

    def test_format_json(self):
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test log message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test_logger"
        assert parsed["message"] == "Test log message"
        assert parsed["service"] == "llm-finetuning"
        assert "timestamp" in parsed

    def test_format_with_extra_fields(self):
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.WARNING,
            pathname="test.py",
            lineno=20,
            msg="Warning message",
            args=(),
            exc_info=None,
        )
        record.custom_metric = 42.5

        formatted = formatter.format(record)
        parsed = json.loads(formatted)
        assert parsed["custom_metric"] == 42.5


class TestColoredConsoleFormatter:
    """Tests for ColoredConsoleFormatter."""

    def test_format_colored(self):
        formatter = ColoredConsoleFormatter(fmt="%(levelname)s - %(message)s")
        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=15,
            msg="Error occurred",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        assert "Error occurred" in formatted


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_instance(self):
        logger = get_logger("my_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "my_module"

    def test_setup_logging(self, tmp_path):
        log_dir = tmp_path / "logs"
        logger = setup_logging(
            log_dir=str(log_dir),
            level="DEBUG",
            format_type="json",
        )
        assert isinstance(logger, logging.Logger)
        assert log_dir.exists()
