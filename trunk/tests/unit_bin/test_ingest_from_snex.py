"""Tests for bin/ingest_from_snex.py using runpy.run_path for real coverage."""
import os
import sys
import types
import runpy
import pytest
from unittest.mock import patch, MagicMock

BIN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'bin')


@pytest.fixture(autouse=True)
def set_lcosndir(monkeypatch, tmp_path):
    monkeypatch.setenv('LCOSNDIR', str(tmp_path))


@pytest.fixture(autouse=True)
def mock_lcogting_module():
    """Ensure LCOGTingest is a mock module in sys.modules before runpy."""
    mock_mod = types.ModuleType('LCOGTingest')
    mock_mod.db_ingest = MagicMock(return_value={})
    sys.modules['LCOGTingest'] = mock_mod
    yield mock_mod
    sys.modules.pop('LCOGTingest', None)


class TestIngestFromSnex:

    def test_single_frame_no_force(self, monkeypatch, tmp_path, mock_lcogting_module):
        """Single frame, no force -> db_ingest called, ingestredu with force='no'."""
        frame = tmp_path / 'img.fits'
        frame.write_text('')  # create empty file

        monkeypatch.setattr('sys.argv', ['ingest_from_snex.py', str(frame)])

        with patch('lsc.mysqldef.ingestredu') as mock_ingestredu:
            runpy.run_path(os.path.join(BIN_DIR, 'ingest_from_snex.py'), run_name='__main__')

        mock_lcogting_module.db_ingest.assert_called_once()
        call_args = mock_lcogting_module.db_ingest.call_args
        assert call_args[1]['force'] is False
        mock_ingestredu.assert_called_once()
        assert mock_ingestredu.call_args[1]['force'] == 'no'

    def test_multiple_frames_with_force(self, monkeypatch, tmp_path, mock_lcogting_module):
        """Multiple frames with --force -> db_ingest called for each, ingestredu force='yes'."""
        frames = []
        for i in range(3):
            f = tmp_path / f'img{i}.fits'
            f.write_text('')
            frames.append(str(f))

        monkeypatch.setattr('sys.argv', ['ingest_from_snex.py', '--force'] + frames)

        with patch('lsc.mysqldef.ingestredu') as mock_ingestredu:
            runpy.run_path(os.path.join(BIN_DIR, 'ingest_from_snex.py'), run_name='__main__')

        assert mock_lcogting_module.db_ingest.call_count == 3
        # All calls should have force=True
        for call in mock_lcogting_module.db_ingest.call_args_list:
            assert call[1]['force'] is True
        mock_ingestredu.assert_called_once()
        assert mock_ingestredu.call_args[1]['force'] == 'yes'

    def test_no_frames_arg_exits(self, monkeypatch):
        """No frames argument -> argparse error, SystemExit."""
        monkeypatch.setattr('sys.argv', ['ingest_from_snex.py'])

        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path(os.path.join(BIN_DIR, 'ingest_from_snex.py'), run_name='__main__')
        assert exc_info.value.code != 0
