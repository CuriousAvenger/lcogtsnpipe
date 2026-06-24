"""
Tests for pure / lightweight utility functions in lsc.util.
Tested: userinput (mocked), readlist (text file).
Heavier FITS-I/O functions are in test_util_io.py.
"""
import pytest
from unittest.mock import patch

pytestmark = pytest.mark.unit


class TestUserinput:
    def test_returns_user_input(self):
        from lsc.util import userinput
        with patch("builtins.input", return_value="hello"):
            assert userinput("prompt: ") == "hello"

    def test_passes_prompt_to_input(self):
        from lsc.util import userinput
        captured = []
        with patch("builtins.input", side_effect=lambda p: captured.append(p) or ""):
            userinput("enter value: ")
        assert "enter value:" in captured[0]

    def test_returns_empty_string(self):
        from lsc.util import userinput
        with patch("builtins.input", return_value=""):
            assert userinput("") == ""


class TestReadlistPure:
    def test_comma_separated_fits_paths(self, simple_fits):
        from lsc.util import readlist
        result = readlist(simple_fits + "," + simple_fits)
        assert len(result) == 2
        assert all(simple_fits == r for r in result)

    def test_single_path_to_fits(self, simple_fits):
        from lsc.util import readlist
        result = readlist(simple_fits)
        assert len(result) == 1
        assert result[0] == simple_fits

    def test_text_file_with_fits_paths(self, simple_fits, tmp_path):
        from lsc.util import readlist
        list_path = tmp_path / "list.txt"
        list_path.write_text(simple_fits + "\n")
        result = readlist(str(list_path))
        assert simple_fits in result

    def test_comments_and_blank_lines_skipped(self, simple_fits, tmp_path):
        from lsc.util import readlist
        list_path = tmp_path / "list2.txt"
        list_path.write_text(f"# comment\n\n{simple_fits}\n")
        result = readlist(str(list_path))
        assert len(result) == 1
        assert result[0] == simple_fits


class TestDefswarp:
    def test_defswarp_without_gain_ron(self, tmp_path):
        """defswarp with gain='' and ron='' → runs without error, writes output file."""
        from lsc.util import defswarp
        out = tmp_path / 'test.swarp'
        result = defswarp(str(out), 'out.fits', 'median', gain='', ron='')
        assert result == str(out)
        assert out.exists()

    def test_defswarp_with_gain(self, tmp_path):
        """defswarp with gain set → writes GAIN_DEFAULT custom value."""
        from lsc.util import defswarp
        out = tmp_path / 'test2.swarp'
        result = defswarp(str(out), 'out.fits', 'average', gain='2.5', ron='')
        assert result == str(out)
        content = out.read_text()
        assert '2.5' in content
