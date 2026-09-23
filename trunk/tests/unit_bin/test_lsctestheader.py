"""Tests for bin/lsctestheader.py — exercises the real script via runpy."""
import os
import sys
import runpy
import pytest
from unittest.mock import patch, MagicMock

BIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'bin'))


@pytest.fixture(autouse=True)
def bin_on_path():
    sys.path.insert(0, BIN_DIR)
    yield
    if BIN_DIR in sys.path:
        sys.path.remove(BIN_DIR)


def _make_readkey3(keyvals):
    """Return a readkey3 mock that returns values from a dict, '' for missing."""
    def fake_readkey3(hdr, key):
        return keyvals.get(key, '')
    return fake_readkey3


class TestLsctestheader:

    def test_with_image_arg_reads_header(self, monkeypatch, capsys):
        """sys.argv has image path -> readhdr called, output printed."""
        monkeypatch.setattr('sys.argv', ['lsctestheader.py', '/data/test.fits'])
        keyvals = {
            'type': 'EXPOSE', 'object': 'SN2024abc', 'mjd': 59000.5,
            'airmass': 1.2, 'filter': 'B', 'grism': '', 'exptime': 300.0,
            'date-obs': '2020-05-01', 'gain': 2.0, 'ron': 10.0,
            'lampid': '', 'RA': 150.0, 'DEC': 2.2, 'datamax': 60000,
            'datamin': -100, 'cenw': '', 'slit': '', 'ut': '12:00:00',
            'NAXIS1': 2048, 'NAXIS2': 2048, 'instrume': 'fa15', 'obsmode': 'NORMAL',
        }
        fake_hdr = MagicMock()
        with patch('lsc.util.readhdr', return_value=fake_hdr) as mock_rh, \
             patch('lsc.util.readkey3', side_effect=_make_readkey3(keyvals)):
            runpy.run_path(os.path.join(BIN_DIR, 'lsctestheader.py'), run_name='__main__')
        mock_rh.assert_called_once_with('/data/test.fits')
        out = capsys.readouterr().out
        assert 'SN2024abc' in out
        assert 'fa15' in out

    def test_all_keywords_present_no_field_fallback(self, monkeypatch, capsys):
        """All keywords present -> field values are real, not fallback placeholders.
        Note: the script has separator lines of '#' and _system is always '#######'."""
        monkeypatch.setattr('sys.argv', ['lsctestheader.py', '/data/test.fits'])
        keyvals = {
            'type': 'EXPOSE', 'object': 'SN2024abc', 'mjd': 59000.5,
            'airmass': 1.2, 'filter': 'r', 'grism': 'OPEN', 'exptime': 120.0,
            'date-obs': '2020-05-01T00:00:00', 'gain': 2.0, 'ron': 5.0,
            'lampid': 'NONE', 'RA': 150.0, 'DEC': 2.2, 'datamax': 60000,
            'datamin': -100, 'cenw': 5500, 'slit': '1.0', 'ut': '12:00:00',
            'NAXIS1': 1024, 'NAXIS2': 1024, 'instrume': 'fl03', 'obsmode': 'NORMAL',
        }
        with patch('lsc.util.readhdr', return_value=MagicMock()), \
             patch('lsc.util.readkey3', side_effect=_make_readkey3(keyvals)):
            runpy.run_path(os.path.join(BIN_DIR, 'lsctestheader.py'), run_name='__main__')
        out = capsys.readouterr().out
        # All real values should appear in output
        assert 'SN2024abc' in out
        assert 'fl03' in out
        assert 'OPEN' in out
        assert '5500' in out

    def test_minimal_header_many_fallbacks(self, monkeypatch, capsys):
        """Minimal header -> many '########' fallbacks."""
        monkeypatch.setattr('sys.argv', ['lsctestheader.py', '/data/test.fits'])
        # Only type and airmass present, rest empty/falsy
        keyvals = {
            'type': 'EXPOSE', 'airmass': 1.0, 'mjd': 59000.0, 'exptime': 60.0,
        }
        with patch('lsc.util.readhdr', return_value=MagicMock()), \
             patch('lsc.util.readkey3', side_effect=_make_readkey3(keyvals)):
            runpy.run_path(os.path.join(BIN_DIR, 'lsctestheader.py'), run_name='__main__')
        out = capsys.readouterr().out
        # Many fields should be ########
        assert '########' in out

    def test_no_argv_uses_glob_and_userinput(self, monkeypatch, capsys):
        """len(sys.argv) <= 1 -> calls glob.glob and userinput."""
        monkeypatch.setattr('sys.argv', ['lsctestheader.py'])
        keyvals = {
            'type': 'EXPOSE', 'object': 'test', 'mjd': 59000.0,
            'airmass': 1.0, 'filter': 'V', 'grism': '', 'exptime': 60.0,
            'date-obs': '2020-01-01', 'gain': 1.0, 'ron': 5.0,
            'lampid': '', 'RA': 0.0, 'DEC': 0.0, 'datamax': 50000,
            'datamin': 0, 'cenw': '', 'slit': '', 'ut': '00:00:00',
            'NAXIS1': 100, 'NAXIS2': 100, 'instrume': 'kb', 'obsmode': '',
        }
        with patch('glob.glob', return_value=['first.fits', 'second.fits']) as mock_glob, \
             patch('lsc.util.userinput', return_value='') as mock_ui, \
             patch('lsc.util.readhdr', return_value=MagicMock()), \
             patch('lsc.util.readkey3', side_effect=_make_readkey3(keyvals)):
            runpy.run_path(os.path.join(BIN_DIR, 'lsctestheader.py'), run_name='__main__')
        mock_glob.assert_called_once_with('*fits')
        mock_ui.assert_called_once()
        out = capsys.readouterr().out
        # Should have printed listing and header info
        assert 'first.fits' in out
