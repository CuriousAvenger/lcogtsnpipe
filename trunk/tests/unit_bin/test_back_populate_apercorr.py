"""Tests for bin/back_populate_apercorr.py using runpy.run_path for real coverage."""
import os
import sys
import runpy
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from astropy.io import fits

BIN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'bin')


@pytest.fixture(autouse=True)
def set_lcosndir(monkeypatch, tmp_path):
    monkeypatch.setenv('LCOSNDIR', str(tmp_path))
    monkeypatch.setattr('lsc.util.workdirectory', str(tmp_path), raising=False)


def _make_file_dict(filenames, filepath):
    return {
        'filename': filenames,
        'filepath': [filepath] * len(filenames),
    }


class TestBackPopulateApercorr:

    def test_sn2_exists_with_apco_updates_db(self, tmp_path):
        """When sn2 file has APCO keyword, updatevalue is called."""
        sn2_path = tmp_path / 'img.sn2.fits'
        hdr = fits.Header()
        hdr['APCO'] = 0.456
        fits.writeto(str(sn2_path), np.zeros((10, 10)), hdr, overwrite=True)

        file_dict = _make_file_dict(['img.fits'], str(tmp_path) + '/')

        with patch('lsc.myloopdef.get_list', return_value=file_dict) as mock_gl, \
             patch('lsc.mysqldef.updatevalue') as mock_update:
            # get_list is called 4 times (filetype 1-4), return data only for first
            mock_gl.side_effect = [file_dict, _make_file_dict([], ''), _make_file_dict([], ''), _make_file_dict([], '')]
            runpy.run_path(os.path.join(BIN_DIR, 'back_populate_apercorr.py'), run_name='__main__')
            mock_update.assert_called_once_with('photlco', 'apercorr', 0.456, 'img.fits')

    def test_sn2_exists_without_apco_no_update(self, tmp_path, capsys):
        """When sn2 file exists but has no APCO, prints message, no updatevalue."""
        sn2_path = tmp_path / 'nokey.sn2.fits'
        hdr = fits.Header()
        fits.writeto(str(sn2_path), np.zeros((10, 10)), hdr, overwrite=True)

        file_dict = _make_file_dict(['nokey.fits'], str(tmp_path) + '/')
        empty = _make_file_dict([], '')

        with patch('lsc.myloopdef.get_list', side_effect=[file_dict, empty, empty, empty]), \
             patch('lsc.mysqldef.updatevalue') as mock_update:
            runpy.run_path(os.path.join(BIN_DIR, 'back_populate_apercorr.py'), run_name='__main__')
            mock_update.assert_not_called()

        captured = capsys.readouterr()
        assert 'APCO not in header' in captured.out

    def test_psf_exists_but_no_sn2_hits_nameerror(self, tmp_path):
        """When .psf.fits exists but .sn2.fits does not, script references undefined sn2_file."""
        psf_path = tmp_path / 'broken.psf.fits'
        fits.writeto(str(psf_path), np.zeros((10, 10)), overwrite=True)

        file_dict = _make_file_dict(['broken.fits'], str(tmp_path) + '/')
        empty = _make_file_dict([], '')

        with patch('lsc.myloopdef.get_list', side_effect=[file_dict, empty, empty, empty]), \
             patch('lsc.mysqldef.updatevalue'):
            # The script has a bug: references undefined `sn2_file` variable
            with pytest.raises(NameError):
                runpy.run_path(os.path.join(BIN_DIR, 'back_populate_apercorr.py'), run_name='__main__')

    def test_get_list_returns_none_raises_typeerror(self, tmp_path):
        """When get_list returns None, iterating over it raises TypeError."""
        with patch('lsc.myloopdef.get_list', return_value=None):
            with pytest.raises(TypeError):
                runpy.run_path(os.path.join(BIN_DIR, 'back_populate_apercorr.py'), run_name='__main__')
