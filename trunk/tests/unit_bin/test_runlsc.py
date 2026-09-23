"""Tests for bin/runlsc.py — exercises __main__ via runpy."""
import sys
import os
import runpy
import importlib
from unittest.mock import patch, MagicMock
import pytest
import numpy as np

TRUNK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(TRUNK, 'bin', 'runlsc.py')


@pytest.fixture
def sub32_mock():
    """Mock subprocess32 module."""
    mock = MagicMock()
    mock.TimeoutExpired = type('TimeoutExpired', (Exception,), {})
    mock.Popen.return_value.wait.return_value = 0
    return mock


def _load_module(sub32_mock):
    """Load runlsc as a module (not __main__) to test run_cmd and commandsn."""
    with patch.dict(sys.modules, {'subprocess32': sub32_mock}):
        loader = importlib.machinery.SourceFileLoader('runlsc', SCRIPT)
        spec = importlib.util.spec_from_loader('runlsc', loader)
        mod = importlib.util.module_from_spec(spec)
        mod.__name__ = 'runlsc'
        spec.loader.exec_module(mod)
    return mod


class TestCommandsn:
    def test_is_dict_with_string_values(self, sub32_mock):
        mod = _load_module(sub32_mock)
        assert isinstance(mod.commandsn, dict)
        for v in mod.commandsn.values():
            assert isinstance(v, str)

    def test_has_known_entries(self, sub32_mock):
        mod = _load_module(sub32_mock)
        assert 'LSQ12fxd' in mod.commandsn
        assert 'SN2013L' in mod.commandsn


class TestRunCmd:
    def test_calls_popen_and_wait(self, sub32_mock, tmp_path):
        mod = _load_module(sub32_mock)
        logf = open(str(tmp_path / 'test.log'), 'w')
        mod.run_cmd('echo hi', logfile=logf, timeout=60)
        logf.close()
        sub32_mock.Popen.assert_called_once()
        sub32_mock.Popen.return_value.wait.assert_called_once_with(60)

    def test_writes_cmd_to_log(self, sub32_mock, tmp_path):
        mod = _load_module(sub32_mock)
        logpath = str(tmp_path / 'cmd.log')
        logf = open(logpath, 'w')
        mod.run_cmd('lscloop.py -s psf', logfile=logf, timeout=100)
        logf.close()
        with open(logpath) as f:
            assert 'lscloop.py -s psf' in f.read()

    def test_timeout_raises(self, sub32_mock, tmp_path):
        mod = _load_module(sub32_mock)
        sub32_mock.Popen.return_value.wait.side_effect = sub32_mock.TimeoutExpired()
        sub32_mock.Popen.return_value.kill = MagicMock()
        logpath = str(tmp_path / 'timeout.log')
        logf = open(logpath, 'w')
        logf.write('some output\n')
        logf.flush()
        # Create the file for reading after close
        with pytest.raises(sub32_mock.TimeoutExpired):
            mod.run_cmd('long_cmd', logfile=logf, timeout=1)


class TestMainBlock:
    """Test the __main__ block via runpy."""

    def test_main_all_telescopes(self, sub32_mock, tmp_path, monkeypatch):
        """Main block with --telescope all should run pipeline for each telescope."""
        monkeypatch.chdir(tmp_path)

        mock_lsc = MagicMock()
        # get_list returns a structured array-like
        mock_list = MagicMock()
        mock_list.__bool__ = lambda s: True
        mock_list.__getitem__ = lambda s, k: {
            'objname': np.array(['SN2024abc']),
            'filepath': np.array(['/data/']),
            'filename': np.array(['img.fits']),
        }[k]
        mock_lsc.myloopdef.get_list.return_value = mock_list
        mock_lsc.util.checksndb.return_value = (None, None, 0)  # not a standard
        mock_lsc.util.getcatalog.return_value = True

        argv = ['runlsc.py', '-e', '20200501', '-T', 'lsc']

        with patch.dict(sys.modules, {'subprocess32': sub32_mock}), \
             patch.object(sys, 'argv', argv), \
             patch.dict(sys.modules, {'lsc': mock_lsc}), \
             patch('lsc.myloopdef', mock_lsc.myloopdef), \
             patch('lsc.util', mock_lsc.util), \
             patch('lsc.mysqldef', mock_lsc.mysqldef):
            runpy.run_path(SCRIPT, run_name='__main__')

        # run_cmd was called multiple times (wcs, psf, zcat, psfmag, mag, fpack)
        sub32_mock.Popen.assert_called()
        assert sub32_mock.Popen.call_count >= 4

    def test_main_with_filter(self, sub32_mock, tmp_path, monkeypatch):
        """Main block with -f flag adds filter to basecmd."""
        monkeypatch.chdir(tmp_path)

        mock_lsc = MagicMock()
        mock_list = MagicMock()
        mock_list.__bool__ = lambda s: True
        mock_list.__getitem__ = lambda s, k: {
            'objname': np.array(['SN2024abc']),
            'filepath': np.array(['/data/']),
            'filename': np.array(['img.fits']),
        }[k]
        mock_lsc.myloopdef.get_list.return_value = mock_list
        mock_lsc.util.checksndb.return_value = (None, None, 0)
        mock_lsc.util.getcatalog.return_value = False

        argv = ['runlsc.py', '-e', '20200501', '-f', 'rp', '-T', 'lsc']

        with patch.dict(sys.modules, {'subprocess32': sub32_mock}), \
             patch.object(sys, 'argv', argv), \
             patch.dict(sys.modules, {'lsc': mock_lsc}), \
             patch('lsc.myloopdef', mock_lsc.myloopdef), \
             patch('lsc.util', mock_lsc.util), \
             patch('lsc.mysqldef', mock_lsc.mysqldef):
            runpy.run_path(SCRIPT, run_name='__main__')

        # Check that 'rp' appears in some Popen call
        calls_str = str(sub32_mock.Popen.call_args_list)
        assert '-f rp' in calls_str

    def test_main_standard_field_skips_psfmag(self, sub32_mock, tmp_path, monkeypatch):
        """Standard fields should be skipped for psfmag/mag stages."""
        monkeypatch.chdir(tmp_path)

        mock_lsc = MagicMock()
        mock_list = MagicMock()
        mock_list.__bool__ = lambda s: True
        mock_list.__getitem__ = lambda s, k: {
            'objname': np.array(['Landolt110']),
            'filepath': np.array(['/data/']),
            'filename': np.array(['std.fits']),
        }[k]
        mock_lsc.myloopdef.get_list.return_value = mock_list
        mock_lsc.util.checksndb.return_value = (None, None, 1)  # objtype=1 => standard
        mock_lsc.util.getcatalog.return_value = True

        argv = ['runlsc.py', '-e', '20200501', '-T', 'lsc']

        with patch.dict(sys.modules, {'subprocess32': sub32_mock}), \
             patch.object(sys, 'argv', argv), \
             patch.dict(sys.modules, {'lsc': mock_lsc}), \
             patch('lsc.myloopdef', mock_lsc.myloopdef), \
             patch('lsc.util', mock_lsc.util), \
             patch('lsc.mysqldef', mock_lsc.mysqldef):
            runpy.run_path(SCRIPT, run_name='__main__')

        # psfmag should not appear for standard fields
        calls_str = str(sub32_mock.Popen.call_args_list)
        assert '-s psfmag' not in calls_str

    def test_main_empty_list(self, sub32_mock, tmp_path, monkeypatch):
        """When get_list returns empty, should skip that telescope."""
        monkeypatch.chdir(tmp_path)

        mock_lsc = MagicMock()
        mock_lsc.myloopdef.get_list.return_value = None  # empty

        argv = ['runlsc.py', '-e', '20200501', '-T', 'lsc']

        with patch.dict(sys.modules, {'subprocess32': sub32_mock}), \
             patch.object(sys, 'argv', argv), \
             patch.dict(sys.modules, {'lsc': mock_lsc}), \
             patch('lsc.myloopdef', mock_lsc.myloopdef), \
             patch('lsc.util', mock_lsc.util), \
             patch('lsc.mysqldef', mock_lsc.mysqldef):
            runpy.run_path(SCRIPT, run_name='__main__')

        # get_list called but no object-specific commands
        mock_lsc.myloopdef.get_list.assert_called()
