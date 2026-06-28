"""Tests for bin/querysloancat.py — thin wrapper around sloan2file."""
import sys
import os
from unittest.mock import patch, MagicMock

TRUNK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIN_DIR = os.path.join(TRUNK_DIR, 'bin')


@patch('lsc.lscabsphotdef.sloan2file')
def test_querysloancat_calls_sloan2file(mock_sloan2file):
    """querysloancat should call sloan2file with parsed arguments."""
    # Simulate what the script does
    ra, dec, radius, mag1, mag2, output = 150.0, 20.0, 10.0, 13.0, 20.0, 'sloan.cat'
    mock_sloan2file(ra, dec, radius, mag1, mag2, output)
    mock_sloan2file.assert_called_once_with(150.0, 20.0, 10.0, 13.0, 20.0, 'sloan.cat')


@patch('lsc.lscabsphotdef.sloan2file')
def test_querysloancat_importable(mock_sloan2file):
    """The script module should be importable without side effects."""
    import importlib
    loader = importlib.machinery.SourceFileLoader('querysloancat', os.path.join(BIN_DIR, 'querysloancat.py'))
    spec = importlib.util.spec_from_loader('querysloancat', loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__name__ = 'querysloancat'
    # This will try to run argparse which calls sys.exit on no args
    # So we patch sys.argv
    with patch('sys.argv', ['querysloancat', '100.0', '25.0', '-r', '15', '-o', 'out.cat']):
        spec.loader.exec_module(mod)
    mock_sloan2file.assert_called_once_with(100.0, 25.0, 15.0, 13.0, 20.0, 'out.cat')
