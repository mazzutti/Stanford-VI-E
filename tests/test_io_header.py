"""Tests for HeaderPrinter formatting and configuration."""

import logging
import pytest

from src.analysis.io import HeaderPrinter, printer


class TestHeaderPrinterInstantiation:
    """Test creating HeaderPrinter instances."""

    def test_multiple_instances_allowed(self):
        """Verify multiple instances can be created."""
        a = HeaderPrinter()
        b = HeaderPrinter()
        c = HeaderPrinter()
        assert a is not b
        assert b is not c
        assert a is not c

    def test_instances_independent(self):
        """Verify instances have independent configuration."""
        a = HeaderPrinter(separator_width=50)
        b = HeaderPrinter(separator_width=100)
        assert a.separator_width == 50
        assert b.separator_width == 100

    def test_module_level_printer(self):
        """Verify module-level printer instance exists."""
        assert isinstance(printer, HeaderPrinter)


class TestHeaderPrinterConfiguration:
    """Test configuration and validation."""

    def test_default_configuration(self):
        """Verify default values are set."""
        hp = HeaderPrinter()
        assert hp.log_level == logging.INFO
        assert hp.separator_width == 70
        assert hp.separator_char == "="

    def test_custom_configuration(self):
        """Verify custom configuration is applied."""
        hp = HeaderPrinter(
            log_level=logging.DEBUG,
            separator_width=80,
            separator_char="-",
        )
        assert hp.log_level == logging.DEBUG
        assert hp.separator_width == 80
        assert hp.separator_char == "-"

    def test_custom_logger(self):
        """Verify custom logger is used."""
        custom_logger = logging.getLogger("custom")
        hp = HeaderPrinter(logger_obj=custom_logger)
        assert hp._logger is custom_logger

    def test_invalid_separator_width_zero(self):
        """Verify ValueError for separator_width=0."""
        with pytest.raises(ValueError, match="separator_width must be >= 1"):
            HeaderPrinter(separator_width=0)

    def test_invalid_separator_width_negative(self):
        """Verify ValueError for negative separator_width."""
        with pytest.raises(ValueError, match="separator_width must be >= 1"):
            HeaderPrinter(separator_width=-5)

    def test_empty_separator_char(self):
        """Verify ValueError for empty separator_char."""
        with pytest.raises(ValueError, match="separator_char cannot be empty"):
            HeaderPrinter(separator_char="")

    def test_configuration_immutability(self):
        """Verify configuration is immutable (properties are read-only)."""
        hp = HeaderPrinter(separator_width=50)
        with pytest.raises(AttributeError):
            hp.separator_width = 100


class TestHeaderPrinterCallable:
    """Test __call__ shorthand."""

    def test_call_shorthand(self, caplog):
        """Verify __call__ works as shorthand for print_analysis_header."""
        hp = HeaderPrinter(separator_width=5)
        with caplog.at_level(logging.INFO):
            hp("TITLE", ["Detail"])

        messages = [record.message for record in caplog.records]
        assert "=====" in messages
        assert "TITLE" in messages
        assert "Detail" in messages

    def test_call_with_none_description(self, caplog):
        """Verify __call__ with None description."""
        hp = HeaderPrinter()
        with caplog.at_level(logging.INFO):
            hp("TEST")

        messages = [record.message for record in caplog.records]
        assert "TEST" in messages


class TestHeaderPrinterFactories:
    """Test factory methods for presets."""

    def test_error_header_factory(self):
        """Verify error_header factory creates correct configuration."""
        hp = HeaderPrinter.error_header()
        assert hp.log_level == logging.WARNING
        assert hp.separator_char == "!"
        assert hp.separator_width == 70

    def test_error_header_logs_at_warning_level(self, caplog):
        """Verify error_header logs at WARNING level."""
        hp = HeaderPrinter.error_header()
        with caplog.at_level(logging.WARNING):
            hp("ERROR", ["Critical issue"])

        assert any(record.levelno == logging.WARNING for record in caplog.records)

    def test_section_header_factory(self):
        """Verify section_header factory creates correct configuration."""
        hp = HeaderPrinter.section_header()
        assert hp.log_level == logging.INFO
        assert hp.separator_char == "-"
        assert hp.separator_width == 50

    def test_error_header_with_custom_logger(self):
        """Verify factory methods accept custom logger."""
        custom_logger = logging.getLogger("error_logger")
        hp = HeaderPrinter.error_header(logger_obj=custom_logger)
        assert hp._logger is custom_logger

    def test_section_header_with_custom_logger(self):
        """Verify section_header accepts custom logger."""
        custom_logger = logging.getLogger("section_logger")
        hp = HeaderPrinter.section_header(logger_obj=custom_logger)
        assert hp._logger is custom_logger

    def test_info_header_factory(self):
        """Verify info_header factory creates correct configuration."""
        hp = HeaderPrinter.info_header()
        assert hp.log_level == logging.INFO
        assert hp.separator_char == "*"
        assert hp.separator_width == 70

    def test_debug_header_factory(self):
        """Verify debug_header factory creates correct configuration."""
        hp = HeaderPrinter.debug_header()
        assert hp.log_level == logging.DEBUG
        assert hp.separator_char == "~"
        assert hp.separator_width == 70

    def test_debug_header_logs_at_debug_level(self, caplog):
        """Verify debug_header logs at DEBUG level."""
        hp = HeaderPrinter.debug_header()
        with caplog.at_level(logging.DEBUG):
            hp("DEBUG", ["Message"])

        assert any(record.levelno == logging.DEBUG for record in caplog.records)


class TestHeaderPrinterFormatting:
    """Test header formatting and logging."""

    def test_separator_generation(self):
        """Verify separator generation."""
        hp = HeaderPrinter(separator_width=5, separator_char="-")
        assert hp._make_separator() == "-----"

    def test_separator_generation_custom_char(self):
        """Verify separator with custom character."""
        hp = HeaderPrinter(separator_width=3, separator_char="*")
        assert hp._make_separator() == "***"

    def test_print_header_with_title_only(self, caplog):
        """Verify header printing with title only."""
        hp = HeaderPrinter(separator_width=5)
        with caplog.at_level(logging.INFO):
            hp.print_analysis_header("TEST")

        messages = [record.message for record in caplog.records]
        assert "=====" in messages
        assert "TEST" in messages
        assert "" in messages  # Blank lines

    def test_print_header_with_description(self, caplog):
        """Verify header printing with title and description."""
        hp = HeaderPrinter(separator_width=5)
        with caplog.at_level(logging.INFO):
            hp.print_analysis_header("TITLE", ["Line 1", "Line 2"])

        messages = [record.message for record in caplog.records]
        assert "=====" in messages
        assert "TITLE" in messages
        assert "Line 1" in messages
        assert "Line 2" in messages

    def test_print_header_with_none_description(self, caplog):
        """Verify header printing with None description."""
        hp = HeaderPrinter()
        with caplog.at_level(logging.INFO):
            hp.print_analysis_header("TITLE", None)

        messages = [record.message for record in caplog.records]
        assert "TITLE" in messages
        assert "=" * 70 in messages

    def test_print_header_with_empty_description(self, caplog):
        """Verify header printing with empty description list."""
        hp = HeaderPrinter(separator_width=5)
        with caplog.at_level(logging.INFO):
            hp.print_analysis_header("TITLE", [])

        messages = [record.message for record in caplog.records]
        assert "=====" in messages
        assert "TITLE" in messages

    def test_print_with_custom_log_level(self, caplog):
        """Verify header prints at custom log level."""
        hp = HeaderPrinter(log_level=logging.WARNING)
        with caplog.at_level(logging.WARNING):
            hp.print_analysis_header("TEST", ["Detail"])

        # Check that messages were logged at WARNING level
        assert any(record.levelno == logging.WARNING for record in caplog.records)


class TestHeaderPrinterFormatLines:
    """Test format_lines generator method."""

    def test_format_lines_with_title_only(self):
        """Verify format_lines generates correct output."""
        hp = HeaderPrinter(separator_width=5)
        lines = list(hp.format_lines("TITLE"))
        assert lines[0] == "====="
        assert lines[1] == "TITLE"
        assert lines[2] == "====="
        assert lines[3] == ""

    def test_format_lines_with_description(self):
        """Verify format_lines with description."""
        hp = HeaderPrinter(separator_width=5)
        lines = list(hp.format_lines("TITLE", ["Line 1", "Line 2"]))
        assert lines[0] == "====="
        assert lines[1] == "TITLE"
        assert lines[2] == "Line 1"
        assert lines[3] == "====="
        assert lines[4] == ""
        assert lines[5] == "Line 1"
        assert lines[6] == "Line 2"
        assert lines[7] == ""

    def test_format_lines_is_generator(self):
        """Verify format_lines returns a generator."""
        hp = HeaderPrinter()
        gen = hp.format_lines("TEST")
        assert hasattr(gen, "__iter__")
        assert hasattr(gen, "__next__")

    def test_format_lines_with_custom_char(self):
        """Verify format_lines uses custom separator."""
        hp = HeaderPrinter(separator_width=3, separator_char="-")
        lines = list(hp.format_lines("TEST"))
        assert lines[0] == "---"
        assert lines[2] == "---"


class TestHeaderPrinterRepr:
    """Test string representation."""

    def test_repr_includes_configuration(self):
        """Verify __repr__ includes relevant information."""
        hp = HeaderPrinter(separator_width=80, separator_char="*")
        repr_str = repr(hp)
        assert "HeaderPrinter" in repr_str
        assert "separator_width=80" in repr_str
        assert "separator_char='*'" in repr_str
        assert "0x" in repr_str  # Memory address

    def test_repr_default_config(self):
        """Verify __repr__ with default configuration."""
        hp = HeaderPrinter()
        repr_str = repr(hp)
        assert "separator_width=70" in repr_str
        assert "separator_char='='" in repr_str


class TestHeaderPrinterContextManager:
    """Test context manager support."""

    def test_context_manager_enter(self):
        """Verify __enter__ returns self."""
        hp = HeaderPrinter()
        with hp as p:
            assert p is hp

    def test_context_manager_exit(self):
        """Verify __exit__ returns None."""
        hp = HeaderPrinter()
        with hp:
            pass  # Verify it doesn't raise

    def test_context_manager_grouping(self, caplog):
        """Verify context manager can group headers."""
        hp = HeaderPrinter(separator_width=5)
        with caplog.at_level(logging.INFO):
            with hp as printer:
                printer("TITLE1", ["Details1"])
                printer("TITLE2", ["Details2"])

        messages = [record.message for record in caplog.records]
        assert "TITLE1" in messages
        assert "TITLE2" in messages


class TestHeaderPrinterFormatString:
    """Test format_string method."""

    def test_format_string_single_line(self):
        """Verify format_string returns string."""
        hp = HeaderPrinter(separator_width=5)
        result = hp.format_string("TITLE")
        assert isinstance(result, str)
        assert "=====" in result
        assert "TITLE" in result

    def test_format_string_with_description(self):
        """Verify format_string with description."""
        hp = HeaderPrinter(separator_width=5)
        result = hp.format_string("TITLE", ["Line 1", "Line 2"])
        lines = result.split("\n")
        assert "=====" in lines
        assert "TITLE" in lines
        assert "Line 1" in lines
        assert "Line 2" in lines

    def test_format_string_newlines(self):
        """Verify format_string uses newlines."""
        hp = HeaderPrinter(separator_width=3)
        result = hp.format_string("T", ["L"])
        assert "\n" in result

    """Test using multiple instances in different contexts."""

    def test_different_printers_different_widths(self, caplog):
        """Verify different instances with different widths."""
        hp1 = HeaderPrinter(separator_width=10, separator_char="-")
        hp2 = HeaderPrinter(separator_width=20, separator_char="*")

        with caplog.at_level(logging.INFO):
            hp1.print_analysis_header("T1")
            hp2.print_analysis_header("T2")

        messages = [record.message for record in caplog.records]
        assert "----------" in messages
        assert "********************" in messages

    def test_different_printers_different_loggers(self):
        """Verify instances with different loggers."""
        logger1 = logging.getLogger("printer1")
        logger2 = logging.getLogger("printer2")

        hp1 = HeaderPrinter(logger_obj=logger1)
        hp2 = HeaderPrinter(logger_obj=logger2)

        assert hp1._logger is logger1
        assert hp2._logger is logger2
        assert hp1._logger is not hp2._logger

    def test_mixing_factories_and_custom(self, caplog):
        """Verify factories can be mixed with custom instances."""
        error_printer = HeaderPrinter.error_header()
        section_printer = HeaderPrinter.section_header()
        custom_printer = HeaderPrinter(separator_width=60, separator_char="#")

        with caplog.at_level(logging.INFO):
            error_printer("ERROR", ["Problem"])
            section_printer("SECTION", ["Details"])
            custom_printer("CUSTOM", ["Info"])

        messages = [record.message for record in caplog.records]
        assert "!" * 70 in messages
        assert "-" * 50 in messages
        assert "#" * 60 in messages
