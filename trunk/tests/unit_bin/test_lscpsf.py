"""Tests for bin/lscpsf.py — exercises the real script via runpy."""
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


def _run_lscpsf(monkeypatch, argv, readlist_rv=None, ecpsf_rv=(1, 5.0, 0.05),
                path_exists_rv=False, getcatalog_rv='cat.cat', userinput_rv=''):
    """Run bin/lscpsf.py with the given mocks."""
    monkeypatch.setattr('sys.argv', ['lscpsf.py'] + argv)
    mocks = {}
    with patch('lsc.util.readlist', return_value=readlist_rv or ['test.fits']) as m1, \
         patch('lsc.lscpsfdef.ecpsf', return_value=ecpsf_rv) as m2, \
         patch('lsc.mysqldef.updatevalue') as m3, \
         patch('lsc.util.getcatalog', return_value=getcatalog_rv) as m4, \
         patch('os.path.exists', return_value=path_exists_rv) as m5, \
         patch('lsc.banzaicat.make_cat', return_value='banzai.cat') as m6, \
         patch('lsc.util.userinput', return_value=userinput_rv):
        # iraf is already mocked via conftest
        runpy.run_path(os.path.join(BIN_DIR, 'lscpsf.py'), run_name='__main__')
        mocks['readlist'] = m1
        mocks['ecpsf'] = m2
        mocks['updatevalue'] = m3
        mocks['getcatalog'] = m4
        mocks['exists'] = m5
        mocks['make_cat'] = m6
    return mocks


class TestLscpsfNormal:

    def test_psf_computed_and_db_updated(self, monkeypatch):
        """Normal image, PSF doesn't exist, ecpsf returns success."""
        mocks = _run_lscpsf(monkeypatch, ['imglist.txt'],
                            readlist_rv=['test.fits'],
                            ecpsf_rv=(1, 5.0, 0.05),
                            path_exists_rv=False)
        mocks['ecpsf'].assert_called_once()
        calls = mocks['updatevalue'].call_args_list
        psf_call = [c for c in calls if c[0][1] == 'psf']
        assert len(psf_call) == 1
        assert psf_call[0][0][2] == 'test.psf.fits'

    def test_psf_already_exists_skips(self, monkeypatch, capsys):
        """PSF already exists and --redo not given -> skip."""
        mocks = _run_lscpsf(monkeypatch, ['imglist.txt'],
                            readlist_rv=['test.fits'],
                            path_exists_rv=True)
        mocks['ecpsf'].assert_not_called()
        assert 'psf already calculated' in capsys.readouterr().out

    def test_redo_flag_recomputes(self, monkeypatch):
        """With --redo, even existing PSF gets recomputed."""
        mocks = _run_lscpsf(monkeypatch, ['imglist.txt', '-r'],
                            readlist_rv=['test.fits'],
                            ecpsf_rv=(1, 4.0, 0.02),
                            path_exists_rv=True)
        mocks['ecpsf'].assert_called_once()

    def test_optimal_in_name_uses_zogypsf(self, monkeypatch):
        """Image with 'optimal' in name uses .zogypsf and psfstars=1."""
        mocks = _run_lscpsf(monkeypatch, ['imglist.txt'],
                            readlist_rv=['test.optimal.fits'],
                            ecpsf_rv=(1, 5.0, 0.01),
                            path_exists_rv=False)
        ecpsf_args = mocks['ecpsf'].call_args[0]
        assert 'zogypsf' in ecpsf_args[0]
        assert ecpsf_args[3] == 1

    def test_optimal_without_fwhm_defaults_to_5(self, monkeypatch):
        """Optimal image without --fwhm defaults fwhm0=5."""
        mocks = _run_lscpsf(monkeypatch, ['imglist.txt'],
                            readlist_rv=['test.optimal.fits'],
                            ecpsf_rv=(1, 5.0, 0.01),
                            path_exists_rv=False)
        ecpsf_args = mocks['ecpsf'].call_args[0]
        # fwhm0 is second arg
        assert ecpsf_args[1] == 5

    def test_optimal_with_fwhm_uses_option(self, monkeypatch):
        """Optimal image with --fwhm uses the provided value."""
        mocks = _run_lscpsf(monkeypatch, ['imglist.txt', '-f', '3.5'],
                            readlist_rv=['test.optimal.fits'],
                            ecpsf_rv=(1, 3.5, 0.01),
                            path_exists_rv=False)
        ecpsf_args = mocks['ecpsf'].call_args[0]
        assert ecpsf_args[1] == 3.5

    def test_ecpsf_returns_0_updates_X(self, monkeypatch):
        """ecpsf returns 0 (bad PSF) -> updatevalue sets psf='X'."""
        mocks = _run_lscpsf(monkeypatch, ['imglist.txt'],
                            readlist_rv=['test.fits'],
                            ecpsf_rv=(0, 3.0, 0.1),
                            path_exists_rv=False)
        calls = mocks['updatevalue'].call_args_list
        psf_call = [c for c in calls if c[0][1] == 'psf']
        assert psf_call[0][0][2] == 'X'

    def test_use_sextractor_option(self, monkeypatch):
        """--use-sextractor sets catalog to empty string."""
        mocks = _run_lscpsf(monkeypatch, ['imglist.txt', '--use-sextractor'],
                            readlist_rv=['test.fits'],
                            ecpsf_rv=(1, 5.0, 0.05),
                            path_exists_rv=False)
        ecpsf_args = mocks['ecpsf'].call_args[0]
        # catalog is the 9th positional arg (index 8)
        assert ecpsf_args[8] == ''

    def test_banzai_option(self, monkeypatch):
        """--banzai calls lsc.banzaicat.make_cat for catalog."""
        mocks = _run_lscpsf(monkeypatch, ['imglist.txt', '--banzai'],
                            readlist_rv=['test.fits'],
                            ecpsf_rv=(1, 5.0, 0.05),
                            path_exists_rv=False)
        mocks['make_cat'].assert_called_once()
        ecpsf_args = mocks['ecpsf'].call_args[0]
        assert ecpsf_args[8] == 'banzai.cat'

    def test_catalog_option(self, monkeypatch):
        """--catalog sets the catalog path directly."""
        mocks = _run_lscpsf(monkeypatch, ['imglist.txt', '-c', '/path/to/my.cat'],
                            readlist_rv=['test.fits'],
                            ecpsf_rv=(1, 5.0, 0.05),
                            path_exists_rv=False)
        ecpsf_args = mocks['ecpsf'].call_args[0]
        assert ecpsf_args[8] == '/path/to/my.cat'

    def test_show_mode_accept_psf(self, monkeypatch):
        """--show mode: user accepts PSF with 'y' -> breaks loop."""
        monkeypatch.setattr('sys.argv', ['lscpsf.py', 'imglist.txt', '-s'])
        with patch('lsc.util.readlist', return_value=['test.fits']), \
             patch('lsc.lscpsfdef.ecpsf', return_value=(1, 5.0, 0.05)), \
             patch('lsc.mysqldef.updatevalue'), \
             patch('lsc.util.getcatalog', return_value='cat.cat'), \
             patch('os.path.exists', return_value=False), \
             patch('lsc.banzaicat.make_cat', return_value='banzai.cat'), \
             patch('lsc.util.userinput', return_value='y'):
            runpy.run_path(os.path.join(BIN_DIR, 'lscpsf.py'), run_name='__main__')

    def test_show_mode_reject_then_enter(self, monkeypatch):
        """--show mode: user rejects with 'n', then presses enter -> breaks."""
        monkeypatch.setattr('sys.argv', ['lscpsf.py', 'imglist.txt', '-s'])
        # First userinput call returns 'n', second returns '' (enter)
        userinput_calls = iter(['n', ''])
        with patch('lsc.util.readlist', return_value=['test.fits']), \
             patch('lsc.lscpsfdef.ecpsf', return_value=(1, 5.0, 0.05)), \
             patch('lsc.mysqldef.updatevalue'), \
             patch('lsc.util.getcatalog', return_value='cat.cat'), \
             patch('os.path.exists', return_value=False), \
             patch('lsc.banzaicat.make_cat', return_value='banzai.cat'), \
             patch('lsc.util.userinput', side_effect=lambda _: next(userinput_calls)):
            runpy.run_path(os.path.join(BIN_DIR, 'lscpsf.py'), run_name='__main__')

    def test_show_mode_reject_new_fwhm(self, monkeypatch):
        """--show mode: user rejects with 'n', provides new fwhm, then accepts."""
        monkeypatch.setattr('sys.argv', ['lscpsf.py', 'imglist.txt', '-s'])
        # First: 'n' (reject), then '4.0' (new fwhm), then 'y' (accept)
        userinput_calls = iter(['n', '4.0', 'y'])
        with patch('lsc.util.readlist', return_value=['test.fits']), \
             patch('lsc.lscpsfdef.ecpsf', return_value=(1, 5.0, 0.05)), \
             patch('lsc.mysqldef.updatevalue'), \
             patch('lsc.util.getcatalog', return_value='cat.cat'), \
             patch('os.path.exists', return_value=False), \
             patch('lsc.banzaicat.make_cat', return_value='banzai.cat'), \
             patch('lsc.util.userinput', side_effect=lambda _: next(userinput_calls)):
            runpy.run_path(os.path.join(BIN_DIR, 'lscpsf.py'), run_name='__main__')

    def test_show_mode_reject_new_datamax(self, monkeypatch):
        """--show mode: user rejects with 'n', provides negative number (datamax), then accepts."""
        monkeypatch.setattr('sys.argv', ['lscpsf.py', 'imglist.txt', '-s'])
        # First: 'n', then '-50000' (new datamax), then 'y' (accept)
        userinput_calls = iter(['n', '-50000', 'y'])
        with patch('lsc.util.readlist', return_value=['test.fits']), \
             patch('lsc.lscpsfdef.ecpsf', return_value=(1, 5.0, 0.05)), \
             patch('lsc.mysqldef.updatevalue'), \
             patch('lsc.util.getcatalog', return_value='cat.cat'), \
             patch('os.path.exists', return_value=False), \
             patch('lsc.banzaicat.make_cat', return_value='banzai.cat'), \
             patch('lsc.util.userinput', side_effect=lambda _: next(userinput_calls)):
            runpy.run_path(os.path.join(BIN_DIR, 'lscpsf.py'), run_name='__main__')

    def test_mysqldef_exception_caught(self, monkeypatch, capsys):
        """If updatevalue raises, the except block prints 'module mysqldef not found'."""
        monkeypatch.setattr('sys.argv', ['lscpsf.py', 'imglist.txt'])
        with patch('lsc.util.readlist', return_value=['test.fits']), \
             patch('lsc.lscpsfdef.ecpsf', return_value=(1, 5.0, 0.05)), \
             patch('lsc.mysqldef.updatevalue', side_effect=Exception('no db')), \
             patch('lsc.util.getcatalog', return_value='cat.cat'), \
             patch('os.path.exists', return_value=False), \
             patch('lsc.banzaicat.make_cat', return_value='banzai.cat'):
            runpy.run_path(os.path.join(BIN_DIR, 'lscpsf.py'), run_name='__main__')
        assert 'module mysqldef not found' in capsys.readouterr().out

    def test_no_positional_arg_shows_help(self, monkeypatch):
        """No positional arg -> SystemExit (help shown)."""
        monkeypatch.setattr('sys.argv', ['lscpsf.py'])
        with patch('lsc.util.readlist', return_value=[]), \
             patch('lsc.lscpsfdef.ecpsf'), \
             patch('lsc.mysqldef.updatevalue'), \
             patch('lsc.util.getcatalog', return_value=''):
            with pytest.raises(SystemExit):
                runpy.run_path(os.path.join(BIN_DIR, 'lscpsf.py'), run_name='__main__')

    def test_invalid_function_shows_help(self, monkeypatch):
        """Invalid --function arg -> SystemExit."""
        monkeypatch.setattr('sys.argv', ['lscpsf.py', '--function', 'bogus', 'imglist.txt'])
        with patch('lsc.util.readlist', return_value=['test.fits']), \
             patch('lsc.lscpsfdef.ecpsf'), \
             patch('lsc.mysqldef.updatevalue'), \
             patch('lsc.util.getcatalog', return_value=''):
            with pytest.raises(SystemExit):
                runpy.run_path(os.path.join(BIN_DIR, 'lscpsf.py'), run_name='__main__')
