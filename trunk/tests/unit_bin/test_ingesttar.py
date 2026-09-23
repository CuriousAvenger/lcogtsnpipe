"""Tests for bin/ingesttar.py — exercises the real script via runpy."""
import os
import sys
import runpy
import tarfile
import pytest
from unittest.mock import patch, MagicMock

BIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'bin'))


@pytest.fixture(autouse=True)
def bin_on_path():
    sys.path.insert(0, BIN_DIR)
    yield
    sys.path.remove(BIN_DIR)


def _make_tar(tmp_path, gz=False):
    """Create a small tar(gz) with a fake file inside."""
    inner = tmp_path / 'img.fits'
    inner.write_bytes(b'\x00' * 64)
    ext = '.tar.gz' if gz else '.tar'
    tar_path = tmp_path / f'snex_data_42{ext}'
    mode = 'w:gz' if gz else 'w'
    with tarfile.open(str(tar_path), mode) as t:
        t.add(str(inner), arcname='img.fits')
    return str(tar_path)


class TestIngesttarRunpy:

    def test_no_file_arg_prints_message(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv('LCOSNDIR', str(tmp_path))
        monkeypatch.setattr('sys.argv', ['ingesttar.py'])
        with patch('lsc.mysqldef.ingestredu'):
            runpy.run_path(os.path.join(BIN_DIR, 'ingesttar.py'), run_name='__main__')
        out = capsys.readouterr().out
        assert 'tar file not included' in out

    def test_with_tar_file_calls_ingestredu_no_force(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv('LCOSNDIR', str(tmp_path))
        tar_path = _make_tar(tmp_path, gz=False)
        monkeypatch.setattr('sys.argv', ['ingesttar.py', '-f', tar_path])
        with patch('lsc.mysqldef.ingestredu') as mock_ingest:
            runpy.run_path(os.path.join(BIN_DIR, 'ingesttar.py'), run_name='__main__')
        mock_ingest.assert_called_once()
        args = mock_ingest.call_args[0]
        assert args[1] == 'no'
        # Check file was extracted
        assert os.path.exists(os.path.join(str(tmp_path), 'img.fits'))

    def test_with_tar_gz_and_force(self, monkeypatch, tmp_path):
        monkeypatch.setenv('LCOSNDIR', str(tmp_path))
        tar_path = _make_tar(tmp_path, gz=True)
        monkeypatch.setattr('sys.argv', ['ingesttar.py', '-f', tar_path, '-G'])
        with patch('lsc.mysqldef.ingestredu') as mock_ingest:
            runpy.run_path(os.path.join(BIN_DIR, 'ingesttar.py'), run_name='__main__')
        mock_ingest.assert_called_once()
        args = mock_ingest.call_args[0]
        assert args[1] == 'yes'

    def test_targetid_from_gz_filename(self, monkeypatch, tmp_path):
        """Target ID parsing from .tar.gz name."""
        monkeypatch.setenv('LCOSNDIR', str(tmp_path))
        # Create tar with specific name pattern
        inner = tmp_path / 'x.fits'
        inner.write_bytes(b'\x00')
        tar_path = tmp_path / 'prefix_99.tar.gz'
        with tarfile.open(str(tar_path), 'w:gz') as t:
            t.add(str(inner), arcname='x.fits')
        monkeypatch.setattr('sys.argv', ['ingesttar.py', '-f', str(tar_path)])
        with patch('lsc.mysqldef.ingestredu') as mock_ingest:
            runpy.run_path(os.path.join(BIN_DIR, 'ingesttar.py'), run_name='__main__')
        # Script should not crash — targetid=99 extracted internally
        mock_ingest.assert_called_once()

    def test_ingesttar_function_directly(self, monkeypatch, tmp_path):
        """Test the ingesttar function imported from the module."""
        monkeypatch.setenv('LCOSNDIR', str(tmp_path))
        tar_path = _make_tar(tmp_path, gz=False)
        with patch('lsc.mysqldef.ingestredu') as mock_ingest:
            ns = runpy.run_path(os.path.join(BIN_DIR, 'ingesttar.py'), run_name='not_main')
            # _dir is set at module level
            ns['_dir'] = str(tmp_path)
            ns['ingesttar'](tar_path, force=True)
        mock_ingest.assert_called_once()
        assert mock_ingest.call_args[0][1] == 'yes'
