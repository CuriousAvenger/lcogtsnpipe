"""Tests for bin/ingestall.py using runpy.run_path for real coverage."""
import os
import sys
import types
import runpy
import base64
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

BIN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'bin')


@pytest.fixture(autouse=True)
def set_lcosndir(monkeypatch, tmp_path):
    monkeypatch.setenv('LCOSNDIR', str(tmp_path))


@pytest.fixture(autouse=True)
def mock_lcogting_module():
    """Mock LCOGTingest module with all functions used by ingestall.py via 'from LCOGTingest import *'."""
    mock_mod = types.ModuleType('LCOGTingest')
    mock_mod.authenticate = MagicMock(return_value='fake-token')
    mock_mod.get_metadata = MagicMock(return_value=[])
    mock_mod.download_frame = MagicMock(return_value=('/data/', 'img.fits'))
    mock_mod.db_ingest = MagicMock(return_value={})
    mock_mod.fits2png = MagicMock()
    mock_mod.record_floyds_tar_link = MagicMock()
    mock_mod.__all__ = ['authenticate', 'get_metadata', 'download_frame', 'db_ingest', 'fits2png', 'record_floyds_tar_link']
    sys.modules['LCOGTingest'] = mock_mod
    yield mock_mod
    sys.modules.pop('LCOGTingest', None)


@pytest.fixture(autouse=True)
def patch_decodestring():
    """base64.decodestring was removed in Python 3.9; patch it back for the script."""
    needs_patch = not hasattr(base64, 'decodestring')
    if needs_patch:
        base64.decodestring = base64.decodebytes
    yield
    if needs_patch:
        del base64.decodestring


class TestIngestallExplicitDaterange:

    def test_explicit_daterange_full_flow(self, monkeypatch, tmp_path, mock_lcogting_module):
        """With explicit daterange, authenticates, queries metadata, downloads, ingests."""
        monkeypatch.setattr('sys.argv', ['ingestall.py', '20200101-20200107'])

        login_data = [{'username': 'user', 'userpw': base64.encodebytes(b'pass')}]

        frame = {'filename': 'coj1m011-fa12-20200101-0001-e91.fits'}
        metadata_returns = [[frame]] + [[] for _ in range(30)]

        mock_lcogting_module.get_metadata.side_effect = metadata_returns
        mock_lcogting_module.download_frame.return_value = ('/data/', 'coj1m011-fa12-20200101-0001-e91.fits')

        with patch('lsc.mysqldef.query', return_value=login_data), \
             patch('lsc.mysqldef.ingestredu') as mock_ingestredu, \
             patch('lsc.myloopdef.conn', MagicMock()):
            runpy.run_path(os.path.join(BIN_DIR, 'ingestall.py'), run_name='__main__')

        mock_lcogting_module.authenticate.assert_called_once()
        mock_lcogting_module.download_frame.assert_called_once_with(frame)
        mock_lcogting_module.db_ingest.assert_called_once_with('/data/', 'coj1m011-fa12-20200101-0001-e91.fits')
        mock_ingestredu.assert_called_once()
        assert '/data/coj1m011-fa12-20200101-0001-e91.fits' in mock_ingestredu.call_args[0][0]


class TestIngestallDefaultDaterange:

    def test_default_daterange_uses_last_week(self, monkeypatch, tmp_path, mock_lcogting_module):
        """No arg -> uses last 7 days as daterange."""
        monkeypatch.setattr('sys.argv', ['ingestall.py'])

        login_data = [{'username': 'user', 'userpw': base64.encodebytes(b'pass')}]
        mock_lcogting_module.get_metadata.return_value = []

        with patch('lsc.mysqldef.query', return_value=login_data), \
             patch('lsc.mysqldef.ingestredu'), \
             patch('lsc.myloopdef.conn', MagicMock()):
            runpy.run_path(os.path.join(BIN_DIR, 'ingestall.py'), run_name='__main__')

        mock_lcogting_module.authenticate.assert_called_once()
        first_call = mock_lcogting_module.get_metadata.call_args_list[0]
        start_kwarg = first_call[1]['start']
        expected_start = datetime.strftime(datetime.utcnow() - timedelta(days=7), '%Y-%m-%d')
        assert start_kwarg == expected_start


class TestIngestallAuthFailure:

    def test_authenticate_raises_valueerror_exits(self, monkeypatch, tmp_path, mock_lcogting_module):
        """When authenticate raises ValueError, script calls sys.exit()."""
        monkeypatch.setattr('sys.argv', ['ingestall.py', '20200101-20200107'])

        login_data = [{'username': 'user', 'userpw': base64.encodebytes(b'pass')}]
        mock_lcogting_module.authenticate.side_effect = ValueError('throttled')

        with patch('lsc.mysqldef.query', return_value=login_data), \
             patch('lsc.myloopdef.conn', MagicMock()):
            with pytest.raises(SystemExit):
                runpy.run_path(os.path.join(BIN_DIR, 'ingestall.py'), run_name='__main__')


class TestIngestallDownloadFailure:

    def test_download_frame_exception_continues(self, monkeypatch, tmp_path, mock_lcogting_module):
        """When download_frame raises, script continues to next frame (no crash)."""
        monkeypatch.setattr('sys.argv', ['ingestall.py', '20200101-20200107'])

        login_data = [{'username': 'user', 'userpw': base64.encodebytes(b'pass')}]
        frames = [
            {'filename': 'fail.fits'},
            {'filename': 'ok-fa12-20200101-0001-e91.fits'},
        ]
        metadata_returns = [frames] + [[] for _ in range(30)]
        mock_lcogting_module.get_metadata.side_effect = metadata_returns

        mock_lcogting_module.download_frame.side_effect = [
            Exception('network error'),
            ('/data/', 'ok-fa12-20200101-0001-e91.fits'),
        ]

        with patch('lsc.mysqldef.query', return_value=login_data), \
             patch('lsc.mysqldef.ingestredu') as mock_ingestredu, \
             patch('lsc.myloopdef.conn', MagicMock()):
            runpy.run_path(os.path.join(BIN_DIR, 'ingestall.py'), run_name='__main__')

        mock_lcogting_module.db_ingest.assert_called_once_with('/data/', 'ok-fa12-20200101-0001-e91.fits')
        mock_ingestredu.assert_called_once()
        assert '/data/ok-fa12-20200101-0001-e91.fits' in mock_ingestredu.call_args[0][0]


class TestIngestallDbIngestFailure:

    def test_db_ingest_exception_continues(self, monkeypatch, tmp_path, mock_lcogting_module):
        """When db_ingest raises, script continues to next frame."""
        monkeypatch.setattr('sys.argv', ['ingestall.py', '20200101-20200107'])

        login_data = [{'username': 'user', 'userpw': base64.encodebytes(b'pass')}]
        frames = [
            {'filename': 'img1-fa12-20200101-0001-e91.fits'},
            {'filename': 'img2-fa12-20200101-0002-e91.fits'},
        ]
        metadata_returns = [frames] + [[] for _ in range(30)]
        mock_lcogting_module.get_metadata.side_effect = metadata_returns

        mock_lcogting_module.download_frame.side_effect = [
            ('/data/', 'img1-fa12-20200101-0001-e91.fits'),
            ('/data/', 'img2-fa12-20200101-0002-e91.fits'),
        ]
        mock_lcogting_module.db_ingest.side_effect = [Exception('db error'), {}]

        with patch('lsc.mysqldef.query', return_value=login_data), \
             patch('lsc.mysqldef.ingestredu'), \
             patch('lsc.myloopdef.conn', MagicMock()):
            runpy.run_path(os.path.join(BIN_DIR, 'ingestall.py'), run_name='__main__')


class TestIngestallSpectraAndPng:

    def test_spectra_excluded_from_fullpaths_and_fits2png_called(self, monkeypatch, tmp_path, mock_lcogting_module):
        """Spectra frames ('-en') excluded from fullpaths; e00 spectra trigger fits2png."""
        monkeypatch.setattr('sys.argv', ['ingestall.py', '20200101-20200107'])

        login_data = [{'username': 'user', 'userpw': base64.encodebytes(b'pass')}]
        frames = [
            {'filename': 'coj1m011-en06-20200101-0001-e00.fits'},
        ]
        metadata_returns = [frames] + [[] for _ in range(30)]
        mock_lcogting_module.get_metadata.side_effect = metadata_returns
        mock_lcogting_module.download_frame.return_value = ('/data/', 'coj1m011-en06-20200101-0001-e00.fits')

        with patch('lsc.mysqldef.query', return_value=login_data), \
             patch('lsc.mysqldef.ingestredu') as mock_ingestredu, \
             patch('lsc.myloopdef.conn', MagicMock()):
            runpy.run_path(os.path.join(BIN_DIR, 'ingestall.py'), run_name='__main__')

        # fits2png should be called for -en + -e00 frames
        mock_lcogting_module.fits2png.assert_called_once()
        # fullpaths should be empty (spectra excluded)
        assert mock_ingestredu.call_args[0][0] == []

    def test_fits2png_exception_continues(self, monkeypatch, tmp_path, mock_lcogting_module):
        """When fits2png raises, script continues."""
        monkeypatch.setattr('sys.argv', ['ingestall.py', '20200101-20200107'])

        login_data = [{'username': 'user', 'userpw': base64.encodebytes(b'pass')}]
        frames = [
            {'filename': 'coj1m011-en06-20200101-0001-e00.fits'},
        ]
        metadata_returns = [frames] + [[] for _ in range(30)]
        mock_lcogting_module.get_metadata.side_effect = metadata_returns
        mock_lcogting_module.download_frame.return_value = ('/data/', 'coj1m011-en06-20200101-0001-e00.fits')
        mock_lcogting_module.fits2png.side_effect = Exception('png error')

        with patch('lsc.mysqldef.query', return_value=login_data), \
             patch('lsc.mysqldef.ingestredu'), \
             patch('lsc.myloopdef.conn', MagicMock()):
            # Should not raise
            runpy.run_path(os.path.join(BIN_DIR, 'ingestall.py'), run_name='__main__')


class TestIngestallFloydsGuider:

    def test_record_floyds_tar_link_exception_continues(self, monkeypatch, tmp_path, mock_lcogting_module):
        """When record_floyds_tar_link raises, script continues."""
        monkeypatch.setattr('sys.argv', ['ingestall.py', '20200101-20200107'])

        login_data = [{'username': 'user', 'userpw': base64.encodebytes(b'pass')}]

        # The script calls get_metadata 18 times for imaging (1 + 5*2 + 6 + 1),
        # then 6 times for guiders (RLEVEL=90).
        guider_frame = {'filename': 'coj1m011-en06-20200101-0001-x90.fits'}
        # Total calls: 18 imaging + 6 guider = 24
        imaging_returns = [[] for _ in range(18)]
        guider_returns = [[guider_frame]] + [[] for _ in range(5)]
        all_returns = imaging_returns + guider_returns

        mock_lcogting_module.get_metadata.side_effect = all_returns
        mock_lcogting_module.record_floyds_tar_link.side_effect = Exception('link error')

        with patch('lsc.mysqldef.query', return_value=login_data), \
             patch('lsc.mysqldef.ingestredu'), \
             patch('lsc.myloopdef.conn', MagicMock()):
            # Should not raise
            runpy.run_path(os.path.join(BIN_DIR, 'ingestall.py'), run_name='__main__')

        mock_lcogting_module.record_floyds_tar_link.assert_called()
