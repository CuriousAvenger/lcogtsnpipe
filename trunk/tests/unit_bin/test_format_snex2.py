"""Tests for bin/format_snex2.py using runpy.run_path for real coverage."""
import os
import sys
import runpy
import builtins
import pytest
from unittest.mock import patch, MagicMock
from astropy.table import Table

SCRIPT_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'bin', 'format_snex2.py')
SCRIPT_PATH = os.path.abspath(SCRIPT_PATH)


@pytest.fixture(autouse=True)
def set_lcosndir(monkeypatch, tmp_path):
    monkeypatch.setenv('LCOSNDIR', str(tmp_path))


class TestFormatSnex2:

    def test_python3_filter_raises_typeerror(self, monkeypatch, tmp_path):
        """In Python 3, filter() returns an iterator; script does [0] on it -> TypeError."""
        infile = tmp_path / 'SN2020abc.dat'
        infile.write_text('SN2020abc 2459000.5 18.5 0.02 18.6 0.03 r\n')

        monkeypatch.setattr('sys.argv', ['format_snex2.py', '-n', str(infile)])

        with pytest.raises(TypeError):
            runpy.run_path(SCRIPT_PATH, run_name='__main__')

    def test_help_exits(self, monkeypatch):
        """--help triggers SystemExit."""
        monkeypatch.setattr('sys.argv', ['format_snex2.py', '--help'])

        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path(SCRIPT_PATH, run_name='__main__')
        assert exc_info.value.code == 0

    def test_no_name_arg_errors(self, monkeypatch):
        """No -n argument -> args.name is None -> ascii.read(None) raises."""
        monkeypatch.setattr('sys.argv', ['format_snex2.py'])

        with pytest.raises((FileNotFoundError, OSError, TypeError, ValueError)):
            runpy.run_path(SCRIPT_PATH, run_name='__main__')

    def test_full_flow_with_patched_filter(self, monkeypatch, tmp_path):
        """Patch builtins.filter to return a list (simulating Python 2 behavior) to cover lines 19-29."""
        infile = tmp_path / 'SN2020abc.dat'
        # Create data with format the script expects:
        # col0=name, col1=JD, col2..col[-3]=mag/err pairs, col[-2] or col[-1]=filter
        infile.write_text('SN2020abc 2459000.5 18.5 0.02 9999. 0. r\n')

        monkeypatch.setattr('sys.argv', ['format_snex2.py', '-n', str(infile)])

        # Monkey-patch filter to return a list so filter(...)[0] works
        original_filter = builtins.filter

        def list_filter(func, iterable):
            return list(original_filter(func, iterable))

        monkeypatch.setattr(builtins, 'filter', list_filter)

        runpy.run_path(SCRIPT_PATH, run_name='__main__')

        # Verify output file was created
        outfile = tmp_path / 'SN2020abc_snex2.csv'
        assert outfile.exists()
        content = outfile.read_text()
        assert 'MJD,filter,mag,error' in content
        assert '59000.0' in content
        assert ',r,' in content
