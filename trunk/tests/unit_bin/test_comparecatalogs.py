"""Tests for bin/comparecatalogs.py using runpy.run_path for real coverage."""
import os
import sys
import runpy
import pytest
from unittest.mock import patch, MagicMock

BIN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'bin')


@pytest.fixture(autouse=True)
def set_lcosndir(monkeypatch, tmp_path):
    monkeypatch.setenv('LCOSNDIR', str(tmp_path))
    monkeypatch.setattr('lsc.util.workdirectory', str(tmp_path), raising=False)


class TestCompareCatalogs:

    def test_default_gaia_no_targets(self, monkeypatch, tmp_path, capsys):
        """Default args, no targets found -> prints count and exits."""
        monkeypatch.setattr('sys.argv', ['comparecatalogs.py', '-f', 'gaia'])

        with patch('lsc.mysqldef.query', return_value=[]) as mock_query, \
             patch('lsc.myloopdef.conn', MagicMock()):
            runpy.run_path(os.path.join(BIN_DIR, 'comparecatalogs.py'), run_name='__main__')

        captured = capsys.readouterr()
        assert '0 targets with no gaia catalog' in captured.out

    def test_catalog_file_exists_on_disk_updates_db(self, monkeypatch, tmp_path):
        """Targets found, catalog file already on disk -> updates DB."""
        monkeypatch.setattr('sys.argv', ['comparecatalogs.py', '-f', 'sloan'])

        catdir = tmp_path / 'standard' / 'cat' / 'sloan'
        catdir.mkdir(parents=True)
        catfile = catdir / 'SN2020abc_sloan.cat'
        catfile.write_text('# fake catalog\n')

        def query_side_effect(qlist, conn):
            q = qlist[0]
            if 'select id, ra0, dec0 from targets' in q:
                return [{'id': 1, 'ra0': 150.0, 'dec0': 2.0}]
            elif 'select name from targetnames' in q:
                return [{'name': 'SN2020abc'}]
            return []

        with patch('lsc.mysqldef.query', side_effect=query_side_effect) as mock_query, \
             patch('lsc.myloopdef.conn', MagicMock()):
            runpy.run_path(os.path.join(BIN_DIR, 'comparecatalogs.py'), run_name='__main__')

        update_calls = [c for c in mock_query.call_args_list if 'update targets set' in c[0][0][0]]
        assert len(update_calls) >= 1
        assert 'SN2020abc_sloan.cat' in update_calls[0][0][0][0]

    def test_no_file_queries_gaia_catalog(self, monkeypatch, tmp_path):
        """Targets found, no file on disk, queries gaia -> gaia2file called."""
        monkeypatch.setattr('sys.argv', ['comparecatalogs.py', '-f', 'gaia'])

        catdir = tmp_path / 'standard' / 'cat' / 'gaia'
        catdir.mkdir(parents=True)

        def query_side_effect(qlist, conn):
            q = qlist[0]
            if 'select id, ra0, dec0 from targets' in q:
                return [{'id': 2, 'ra0': 100.0, 'dec0': -30.0}]
            elif 'select name from targetnames' in q:
                return [{'name': 'SN2021xyz'}]
            return []

        with patch('lsc.mysqldef.query', side_effect=query_side_effect), \
             patch('lsc.myloopdef.conn', MagicMock()), \
             patch('lsc.lscabsphotdef.gaia2file') as mock_gaia:
            runpy.run_path(os.path.join(BIN_DIR, 'comparecatalogs.py'), run_name='__main__')

        mock_gaia.assert_called_once()
        call_kwargs = mock_gaia.call_args
        assert call_kwargs[0][0] == 100.0
        assert call_kwargs[0][1] == -30.0

    def test_force_flag_uses_empty_string_condition(self, monkeypatch, tmp_path, capsys):
        """With --force, queries targets where cat="" instead of is null."""
        monkeypatch.setattr('sys.argv', ['comparecatalogs.py', '-f', 'gaia', '--force'])

        with patch('lsc.mysqldef.query', return_value=[]) as mock_query, \
             patch('lsc.myloopdef.conn', MagicMock()):
            runpy.run_path(os.path.join(BIN_DIR, 'comparecatalogs.py'), run_name='__main__')

        first_call = mock_query.call_args_list[0]
        assert '=""' in first_call[0][0][0]

    def test_panstarrs_flag_for_sloan(self, monkeypatch, tmp_path):
        """With --panstarrs and sloan field, panstarrs2file is called."""
        monkeypatch.setattr('sys.argv', ['comparecatalogs.py', '-f', 'sloan', '--panstarrs'])

        catdir = tmp_path / 'standard' / 'cat' / 'sloan'
        catdir.mkdir(parents=True)

        def query_side_effect(qlist, conn):
            q = qlist[0]
            if 'select id, ra0, dec0 from targets' in q:
                return [{'id': 3, 'ra0': 200.0, 'dec0': 10.0}]
            elif 'select name from targetnames' in q:
                return [{'name': 'SN2022pan'}]
            return []

        with patch('lsc.mysqldef.query', side_effect=query_side_effect), \
             patch('lsc.myloopdef.conn', MagicMock()), \
             patch('lsc.lscabsphotdef.panstarrs2file') as mock_ps:
            runpy.run_path(os.path.join(BIN_DIR, 'comparecatalogs.py'), run_name='__main__')

        mock_ps.assert_called_once()
        assert mock_ps.call_args[0][0] == 200.0
        assert mock_ps.call_args[0][1] == 10.0

    def test_apass_query_uses_os_system(self, monkeypatch, tmp_path):
        """With -f apass, uses os.system for queryapasscat.py."""
        monkeypatch.setattr('sys.argv', ['comparecatalogs.py', '-f', 'apass'])

        catdir = tmp_path / 'standard' / 'cat' / 'apass'
        catdir.mkdir(parents=True)

        def query_side_effect(qlist, conn):
            q = qlist[0]
            if 'select id, ra0, dec0 from targets' in q:
                return [{'id': 4, 'ra0': 180.0, 'dec0': 5.0}]
            elif 'select name from targetnames' in q:
                return [{'name': 'SN2023apass'}]
            return []

        with patch('lsc.mysqldef.query', side_effect=query_side_effect), \
             patch('lsc.myloopdef.conn', MagicMock()), \
             patch('os.system') as mock_system:
            runpy.run_path(os.path.join(BIN_DIR, 'comparecatalogs.py'), run_name='__main__')

        mock_system.assert_called_once()
        assert 'queryapasscat.py' in mock_system.call_args[0][0]
        assert '180.0' in mock_system.call_args[0][0]

    def test_sloan_query_without_panstarrs(self, monkeypatch, tmp_path):
        """With -f sloan (no --panstarrs), sloan2file is called."""
        monkeypatch.setattr('sys.argv', ['comparecatalogs.py', '-f', 'sloan'])

        catdir = tmp_path / 'standard' / 'cat' / 'sloan'
        catdir.mkdir(parents=True)

        def query_side_effect(qlist, conn):
            q = qlist[0]
            if 'select id, ra0, dec0 from targets' in q:
                return [{'id': 5, 'ra0': 120.0, 'dec0': -10.0}]
            elif 'select name from targetnames' in q:
                return [{'name': 'SN2023sloan'}]
            return []

        with patch('lsc.mysqldef.query', side_effect=query_side_effect), \
             patch('lsc.myloopdef.conn', MagicMock()), \
             patch('lsc.lscabsphotdef.sloan2file') as mock_sloan:
            runpy.run_path(os.path.join(BIN_DIR, 'comparecatalogs.py'), run_name='__main__')

        mock_sloan.assert_called_once()
        assert mock_sloan.call_args[0][0] == 120.0
        assert mock_sloan.call_args[0][1] == -10.0

    def test_catalog_query_exception_handled(self, monkeypatch, tmp_path, capsys):
        """Exception during catalog query prints error and continues."""
        monkeypatch.setattr('sys.argv', ['comparecatalogs.py', '-f', 'gaia'])

        catdir = tmp_path / 'standard' / 'cat' / 'gaia'
        catdir.mkdir(parents=True)

        def query_side_effect(qlist, conn):
            q = qlist[0]
            if 'select id, ra0, dec0 from targets' in q:
                return [{'id': 6, 'ra0': 90.0, 'dec0': 45.0}]
            elif 'select name from targetnames' in q:
                return [{'name': 'SN2023err'}]
            return []

        with patch('lsc.mysqldef.query', side_effect=query_side_effect), \
             patch('lsc.myloopdef.conn', MagicMock()), \
             patch('lsc.lscabsphotdef.gaia2file', side_effect=Exception('network fail')):
            runpy.run_path(os.path.join(BIN_DIR, 'comparecatalogs.py'), run_name='__main__')

        out = capsys.readouterr().out
        assert 'Catalog query failed:' in out

    def test_catalog_download_success_updates_db(self, monkeypatch, tmp_path):
        """When catalog download creates the file, DB is updated."""
        monkeypatch.setattr('sys.argv', ['comparecatalogs.py', '-f', 'gaia'])

        catdir = tmp_path / 'standard' / 'cat' / 'gaia'
        catdir.mkdir(parents=True)

        def query_side_effect(qlist, conn):
            q = qlist[0]
            if 'select id, ra0, dec0 from targets' in q:
                return [{'id': 7, 'ra0': 45.0, 'dec0': 20.0}]
            elif 'select name from targetnames' in q:
                return [{'name': 'SN2023dl'}]
            return []

        def gaia2file_creates_file(ra, dec, output=''):
            # Create the file that the script checks for
            with open(output, 'w') as f:
                f.write('# catalog\n')

        with patch('lsc.mysqldef.query', side_effect=query_side_effect) as mock_query, \
             patch('lsc.myloopdef.conn', MagicMock()), \
             patch('lsc.lscabsphotdef.gaia2file', side_effect=gaia2file_creates_file):
            runpy.run_path(os.path.join(BIN_DIR, 'comparecatalogs.py'), run_name='__main__')

        # Should have update query with the filename
        update_calls = [c for c in mock_query.call_args_list
                        if 'update targets set' in c[0][0][0] and 'gaia_cat="SN2023dl_gaia.cat"' in c[0][0][0]]
        assert len(update_calls) >= 1
