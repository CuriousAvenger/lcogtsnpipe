"""
Additional tests for lsc.myloopdef — covers check_missing, checkfilevsdatabase,
and get_standards.
No database or network access required.
"""
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# check_missing — copies missing files from raw to reduced directory
# ---------------------------------------------------------------------------

class TestCheckMissing:
    def test_copies_missing_file(self, tmp_path, monkeypatch):
        """If file exists in raw but not reduced, it should be copied."""
        from lsc.myloopdef import check_missing

        raw_dir = str(tmp_path / 'raw')
        red_dir = str(tmp_path / 'red')
        os.makedirs(raw_dir)
        os.makedirs(red_dir)

        # Create file in raw but not reduced
        fname = 'test.fits'
        with open(os.path.join(raw_dir, fname), 'w') as f:
            f.write('raw data')

        # Mock the DB queries
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [{'filepath': raw_dir + '/'}]
        mock_conn.cursor.return_value = mock_cursor

        with patch('lsc.myloopdef.lsc') as mock_lsc:
            mock_lsc.mysqldef.getfromdataraw.side_effect = [
                [{'filepath': raw_dir + '/'}],      # raw query
                [{'filepath': red_dir + '/'}],       # reduced query
            ]
            with patch('os.path.isfile', side_effect=lambda p: p == os.path.join(raw_dir, fname)):
                with patch('os.system') as mock_system:
                    check_missing([fname], database='photlco')
                    mock_system.assert_called_once()
                    call_args = mock_system.call_args[0][0]
                    assert 'cp' in call_args
                    assert fname in call_args

    def test_no_action_when_file_exists(self, monkeypatch):
        """If file already exists in reduced dir, no copy needed."""
        from lsc.myloopdef import check_missing

        with patch('lsc.myloopdef.lsc') as mock_lsc:
            mock_lsc.mysqldef.getfromdataraw.return_value = [{'filepath': '/some/path/'}]
            with patch('os.path.isfile', return_value=True):
                with patch('os.system') as mock_system:
                    check_missing(['test.fits'])
                    mock_system.assert_not_called()

    def test_empty_list_noop(self, monkeypatch):
        """Empty list should do nothing."""
        from lsc.myloopdef import check_missing
        with patch('lsc.myloopdef.lsc') as mock_lsc:
            check_missing([])
            mock_lsc.mysqldef.getfromdataraw.assert_not_called()


# ---------------------------------------------------------------------------
# checkfilevsdatabase — compares FITS header values with DB values
# ---------------------------------------------------------------------------

class TestCheckfilevsdatabase:
    def test_does_not_update_when_matching(self, monkeypatch, tmp_path):
        """When FITS header and DB match, no update is issued."""
        from lsc.myloopdef import checkfilevsdatabase

        fits_file = str(tmp_path / 'test.sn2.fits')
        # Create minimal FITS file
        from astropy.io import fits
        hdr = fits.Header()
        hdr['FILTER'] = 'r'
        hdr['EXPTIME'] = 120.0
        hdr['AIRMASS'] = 1.2
        hdr['TELESCOP'] = '1m0-01'
        hdr['PSFMAG1'] = 18.5
        hdr['PSFDMAG1'] = 0.05
        hdr['APMAG1'] = 18.4
        hdr['MAG'] = 18.3
        fits.writeto(fits_file, np.ones((10, 10)), header=hdr, overwrite=True)

        lista = {
            'filepath': [str(tmp_path) + '/'],
            'filename': ['test.fits'],
            'mag': [18.3],
            'psfmag': [18.5],
        }

        with patch('lsc.myloopdef.lsc') as mock_lsc:
            mock_lsc.util.readhdr.return_value = hdr
            mock_lsc.util.readkey3.side_effect = lambda h, k: hdr.get(k, '')
            mock_lsc.mysqldef.updatevalue = MagicMock()
            checkfilevsdatabase(lista)
            # MAG matches, so updatevalue should NOT be called for mag
            mock_lsc.mysqldef.updatevalue.assert_not_called()

    def test_updates_when_mismatch(self, monkeypatch, tmp_path):
        """When FITS header differs from DB, update is issued."""
        from lsc.myloopdef import checkfilevsdatabase

        fits_file = str(tmp_path / 'test.sn2.fits')
        from astropy.io import fits
        hdr = fits.Header()
        hdr['FILTER'] = 'r'
        hdr['EXPTIME'] = 120.0
        hdr['AIRMASS'] = 1.2
        hdr['TELESCOP'] = '1m0-01'
        hdr['PSFMAG1'] = 19.0  # Different from DB
        hdr['PSFDMAG1'] = 0.05
        hdr['APMAG1'] = 18.4
        hdr['MAG'] = 19.0  # Different from DB
        fits.writeto(fits_file, np.ones((10, 10)), header=hdr, overwrite=True)

        lista = {
            'filepath': [str(tmp_path) + '/'],
            'filename': ['test.fits'],
            'mag': [18.3],  # DB has 18.3, FITS has 19.0
            'psfmag': [18.5],  # DB has 18.5, FITS has 19.0
        }

        with patch('lsc.myloopdef.lsc') as mock_lsc:
            mock_lsc.util.readhdr.return_value = hdr
            mock_lsc.util.readkey3.side_effect = lambda h, k: hdr.get(k, '')
            mock_lsc.mysqldef.updatevalue = MagicMock()
            checkfilevsdatabase(lista)
            # Should be called for both mag and psfmag mismatches
            assert mock_lsc.mysqldef.updatevalue.call_count >= 2

    def test_empty_list(self):
        """Empty list should do nothing."""
        from lsc.myloopdef import checkfilevsdatabase
        with patch('lsc.myloopdef.lsc') as mock_lsc:
            checkfilevsdatabase({'filepath': [], 'filename': []})
            mock_lsc.util.readhdr.assert_not_called()


# ---------------------------------------------------------------------------
# process_epoch — date range parsing
# ---------------------------------------------------------------------------

class TestProcessEpoch:
    def test_none_returns_tuple_of_two_strings(self):
        from lsc.myloopdef import process_epoch
        result = process_epoch(None)
        assert len(result) == 2
        assert all(isinstance(s, str) for s in result)
        assert len(result[0]) == 8  # YYYYMMDD
        assert len(result[1]) == 8

    def test_single_date(self):
        from lsc.myloopdef import process_epoch
        result = process_epoch('20200101')
        assert result == ['20200101']

    def test_date_range(self):
        from lsc.myloopdef import process_epoch
        result = process_epoch('20200101-20200131')
        assert result == ['20200101', '20200131']

    def test_none_range_is_four_days(self):
        from lsc.myloopdef import process_epoch
        first, second = process_epoch(None)
        # The difference should be ~4 days
        from datetime import datetime
        d1 = datetime.strptime(first, '%Y%m%d')
        d2 = datetime.strptime(second, '%Y%m%d')
        assert 3 <= (d2 - d1).days <= 5
