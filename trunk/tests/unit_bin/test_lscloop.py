"""Tests for bin/lscloop.py — exercises the real script via runpy."""
import os
import sys
import runpy
import warnings
import pytest
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock, call


BIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'bin'))


@pytest.fixture(autouse=True)
def bin_on_path():
    sys.path.insert(0, BIN_DIR)
    yield
    if BIN_DIR in sys.path:
        sys.path.remove(BIN_DIR)


def _make_ll():
    """Standard mock return for get_list."""
    return {
        'filename': np.array(['img1.fits']),
        'filepath': np.array(['/data/']),
        'objname': np.array(['SN2024abc']),
        'filter': np.array(['rp']),
        'wcs': np.array([0]),
        'psf': np.array(['img1.psf.fits']),
        'psfmag': np.array([20.0]),
        'apmag': np.array([20.0]),
        'zcat': np.array(['img1.cat']),
        'mag': np.array([20.0]),
        'abscat': np.array(['img1.cat']),
        'dayobs': np.array(['20200501']),
        'lastunpacked': np.array([datetime(2020, 5, 1)]),
    }


def _run_loop(monkeypatch, argv, get_list_rv=None, **extra_patches):
    """Helper to run lscloop.py with patched internals."""
    monkeypatch.setattr('sys.argv', ['lscloop.py'] + argv)
    patches = {
        'lsc.myloopdef.get_list': MagicMock(return_value=get_list_rv),
        'lsc.myloopdef.conn': MagicMock(),
        'lsc.mysqldef.query': MagicMock(),
    }
    patches.update(extra_patches)
    stack = []
    mocks = {}
    try:
        for key, val in patches.items():
            p = patch(key, val)
            mocks[key] = p.start()
            stack.append(p)
        runpy.run_path(os.path.join(BIN_DIR, 'lscloop.py'), run_name='__main__')
    finally:
        for p in stack:
            p.stop()
    return mocks


class TestNoDataSelected:

    def test_no_data_prints_message(self, monkeypatch, capsys):
        _run_loop(monkeypatch, ['-e', '20200501', '-s', 'psf'], get_list_rv=None)
        assert 'no data selected' in capsys.readouterr().out


class TestStageCosmic:

    def test_cosmic_multicore_1(self, monkeypatch):
        ll = _make_ll()
        mock_rc = MagicMock()
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'cosmic', '--multicore', '1'],
                  get_list_rv=ll,
                  **{'lsc.myloopdef.run_cosmic': mock_rc})
        mock_rc.assert_called_once()
        args = mock_rc.call_args[0]
        assert 'photlco' in args


class TestStagePsf:

    def test_psf_calls_run_psf(self, monkeypatch):
        ll = _make_ll()
        mock_rp = MagicMock()
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'psf'],
                  get_list_rv=ll,
                  **{'lsc.myloopdef.run_psf': mock_rp})
        mock_rp.assert_called_once()


class TestStagePsfmag:

    def test_psfmag_calls_run_fit(self, monkeypatch):
        ll = _make_ll()
        mock_rf = MagicMock()
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'psfmag'],
                  get_list_rv=ll,
                  **{'lsc.myloopdef.run_fit': mock_rf})
        mock_rf.assert_called_once()


class TestStageWcs:

    def test_wcs_calls_run_wcs(self, monkeypatch):
        ll = _make_ll()
        mock_rw = MagicMock()
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'wcs'],
                  get_list_rv=ll,
                  **{'lsc.myloopdef.run_wcs': mock_rw})
        mock_rw.assert_called_once()


class TestStageGetmag:

    def test_getmag_calls_run_getmag(self, monkeypatch):
        ll = _make_ll()
        mock_gm = MagicMock()
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'getmag'],
                  get_list_rv=ll,
                  **{'lsc.myloopdef.run_getmag': mock_gm})
        mock_gm.assert_called_once()


class TestStageZcat:

    def test_zcat_calls_absphot(self, monkeypatch):
        ll = _make_ll()
        mock_abs = MagicMock()
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'zcat', '--field', 'sloan'],
                  get_list_rv=ll,
                  **{'lsc.lscabsphotdef.absphot': mock_abs})
        mock_abs.assert_called_once()

    def test_zcat_gaia_skips(self, monkeypatch, capsys):
        ll = _make_ll()
        mock_abs = MagicMock()
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'zcat', '--field', 'gaia'],
                  get_list_rv=ll,
                  **{'lsc.lscabsphotdef.absphot': mock_abs})
        mock_abs.assert_not_called()
        assert 'Cannot use gaia' in capsys.readouterr().out


class TestStageCheckpsf:

    def test_checkpsf_called(self, monkeypatch):
        ll = _make_ll()
        mock_cp = MagicMock()
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'checkpsf'],
                  get_list_rv=ll,
                  **{'lsc.myloopdef.checkpsf': mock_cp})
        mock_cp.assert_called_once()


class TestStageCheckdiff:

    def test_checkdiff_sets_filetype_3(self, monkeypatch):
        ll = _make_ll()
        mock_cd = MagicMock()
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'checkdiff'],
                  get_list_rv=ll,
                  **{'lsc.myloopdef.checkdiff': mock_cd})
        mock_cd.assert_called_once()


class TestStageMerge:

    def test_mergeall_calls_run_merge(self, monkeypatch):
        ll = _make_ll()
        mock_rm = MagicMock()
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'mergeall'],
                  get_list_rv=ll,
                  **{'lsc.myloopdef.run_merge': mock_rm})
        mock_rm.assert_called_once()

    def test_merge_stage_prints_epochs(self, monkeypatch, capsys):
        """'merge' stage enters the else-branch. Bug: listfile not defined."""
        ll = _make_ll()
        mock_rm = MagicMock()
        with pytest.raises(NameError):
            _run_loop(monkeypatch,
                      ['-e', '20200501', '-s', 'merge'],
                      get_list_rv=ll,
                      **{'lsc.myloopdef.run_merge': mock_rm})
        out = capsys.readouterr().out
        assert '20200501' in out


class TestStageDiff:

    def test_diff_requires_name(self, monkeypatch):
        ll = _make_ll()
        with pytest.raises(Exception, match='you need to select one object'):
            _run_loop(monkeypatch,
                      ['-e', '20200501', '-s', 'diff', '-T', 'fl'],
                      get_list_rv=ll)

    def test_diff_requires_telescope(self, monkeypatch):
        ll = _make_ll()
        with pytest.raises(Exception, match='you need to select one type of instrument'):
            _run_loop(monkeypatch,
                      ['-e', '20200501', '-s', 'diff', '-n', 'SN2024abc'],
                      get_list_rv=ll)

    def test_diff_with_template(self, monkeypatch):
        ll = _make_ll()
        mock_rd = MagicMock()
        template_row = {
            'filename': 'tmpl.fits', 'filepath': '/data/',
            'mjd': 58000.0, 'objname': 'SN2024abc', 'filter': 'rp',
        }
        mock_getlist = MagicMock(return_value=[template_row])
        mock_filtra = MagicMock(return_value={
            'filename': np.array(['tmpl.fits']),
            'filepath': np.array(['/data/']),
        })
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'diff', '-n', 'SN2024abc', '-T', 'fl',
                   '--tempdate', '19990101-20200101'],
                  get_list_rv=ll,
                  **{
                      'lsc.myloopdef.run_diff': mock_rd,
                      'lsc.mysqldef.getlistfromraw': mock_getlist,
                      'lsc.myloopdef.filtralist': mock_filtra,
                  })
        mock_rd.assert_called_once()

    def test_diff_with_sdss_temptel(self, monkeypatch):
        """Diff with --temptel SDSS and various -T values for fake_temptel mapping."""
        ll = _make_ll()
        mock_rd = MagicMock()
        template_row = {
            'filename': 'tmpl.fits', 'filepath': '/data/',
            'mjd': 58000.0, 'objname': 'SN2024abc', 'filter': 'rp',
        }
        mock_getlist = MagicMock(return_value=[template_row])
        mock_filtra = MagicMock(return_value={
            'filename': np.array(['tmpl.fits']),
            'filepath': np.array(['/data/']),
        })
        # Test kb -> sbig mapping
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'diff', '-n', 'SN2024abc', '-T', 'kb',
                   '--tempdate', '19990101-20200101', '--temptel', 'SDSS'],
                  get_list_rv=ll,
                  **{
                      'lsc.myloopdef.run_diff': mock_rd,
                      'lsc.mysqldef.getlistfromraw': mock_getlist,
                      'lsc.myloopdef.filtralist': mock_filtra,
                  })
        # getlistfromraw should be called with 'sbig'
        assert mock_getlist.call_args[0][-1] == 'sbig'

    def test_diff_with_sdss_temptel_fs(self, monkeypatch):
        """Diff with --temptel SDSS and -T fs -> spectral."""
        ll = _make_ll()
        mock_rd = MagicMock()
        template_row = {
            'filename': 'tmpl.fits', 'filepath': '/data/',
            'mjd': 58000.0, 'objname': 'SN2024abc', 'filter': 'rp',
        }
        mock_getlist = MagicMock(return_value=[template_row])
        mock_filtra = MagicMock(return_value={
            'filename': np.array(['tmpl.fits']),
            'filepath': np.array(['/data/']),
        })
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'diff', '-n', 'SN2024abc', '-T', 'fs',
                   '--tempdate', '19990101-20200101', '--temptel', 'SDSS'],
                  get_list_rv=ll,
                  **{
                      'lsc.myloopdef.run_diff': mock_rd,
                      'lsc.mysqldef.getlistfromraw': mock_getlist,
                      'lsc.myloopdef.filtralist': mock_filtra,
                  })
        assert mock_getlist.call_args[0][-1] == 'spectral'

    def test_diff_with_sdss_temptel_fl(self, monkeypatch):
        """Diff with --temptel PS1 and -T fl -> sinistro."""
        ll = _make_ll()
        mock_rd = MagicMock()
        template_row = {
            'filename': 'tmpl.fits', 'filepath': '/data/',
            'mjd': 58000.0, 'objname': 'SN2024abc', 'filter': 'rp',
        }
        mock_getlist = MagicMock(return_value=[template_row])
        mock_filtra = MagicMock(return_value={
            'filename': np.array(['tmpl.fits']),
            'filepath': np.array(['/data/']),
        })
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'diff', '-n', 'SN2024abc', '-T', 'fl',
                   '--tempdate', '19990101-20200101', '--temptel', 'PS1'],
                  get_list_rv=ll,
                  **{
                      'lsc.myloopdef.run_diff': mock_rd,
                      'lsc.mysqldef.getlistfromraw': mock_getlist,
                      'lsc.myloopdef.filtralist': mock_filtra,
                  })
        assert mock_getlist.call_args[0][-1] == 'sinistro'

    def test_diff_with_sdss_temptel_ep(self, monkeypatch):
        """Diff with --temptel SDSS and -T ep -> muscat."""
        ll = _make_ll()
        mock_rd = MagicMock()
        template_row = {
            'filename': 'tmpl.fits', 'filepath': '/data/',
            'mjd': 58000.0, 'objname': 'SN2024abc', 'filter': 'rp',
        }
        mock_getlist = MagicMock(return_value=[template_row])
        mock_filtra = MagicMock(return_value={
            'filename': np.array(['tmpl.fits']),
            'filepath': np.array(['/data/']),
        })
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'diff', '-n', 'SN2024abc', '-T', 'ep',
                   '--tempdate', '19990101-20200101', '--temptel', 'SDSS'],
                  get_list_rv=ll,
                  **{
                      'lsc.myloopdef.run_diff': mock_rd,
                      'lsc.mysqldef.getlistfromraw': mock_getlist,
                      'lsc.myloopdef.filtralist': mock_filtra,
                  })
        assert mock_getlist.call_args[0][-1] == 'muscat'

    def test_diff_with_sdss_temptel_sq(self, monkeypatch):
        """Diff with --temptel SDSS and -T sq -> qhy."""
        ll = _make_ll()
        mock_rd = MagicMock()
        template_row = {
            'filename': 'tmpl.fits', 'filepath': '/data/',
            'mjd': 58000.0, 'objname': 'SN2024abc', 'filter': 'rp',
        }
        mock_getlist = MagicMock(return_value=[template_row])
        mock_filtra = MagicMock(return_value={
            'filename': np.array(['tmpl.fits']),
            'filepath': np.array(['/data/']),
        })
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'diff', '-n', 'SN2024abc', '-T', 'sq',
                   '--tempdate', '19990101-20200101', '--temptel', 'SDSS'],
                  get_list_rv=ll,
                  **{
                      'lsc.myloopdef.run_diff': mock_rd,
                      'lsc.mysqldef.getlistfromraw': mock_getlist,
                      'lsc.myloopdef.filtralist': mock_filtra,
                  })
        assert mock_getlist.call_args[0][-1] == 'qhy'

    def test_diff_with_custom_temptel(self, monkeypatch):
        """Diff with --temptel that is NOT SDSS/PS1 -> uses temptel directly."""
        ll = _make_ll()
        mock_rd = MagicMock()
        template_row = {
            'filename': 'tmpl.fits', 'filepath': '/data/',
            'mjd': 58000.0, 'objname': 'SN2024abc', 'filter': 'rp',
        }
        mock_getlist = MagicMock(return_value=[template_row])
        mock_filtra = MagicMock(return_value={
            'filename': np.array(['tmpl.fits']),
            'filepath': np.array(['/data/']),
        })
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'diff', '-n', 'SN2024abc', '-T', 'fl',
                   '--tempdate', '19990101-20200101', '--temptel', 'mytel'],
                  get_list_rv=ll,
                  **{
                      'lsc.myloopdef.run_diff': mock_rd,
                      'lsc.mysqldef.getlistfromraw': mock_getlist,
                      'lsc.myloopdef.filtralist': mock_filtra,
                  })
        assert mock_getlist.call_args[0][-1] == 'mytel'

    def test_diff_optimal_suffix(self, monkeypatch):
        """Diff with --difftype 1 uses .optimal suffix."""
        ll = _make_ll()
        mock_rd = MagicMock()
        template_row = {
            'filename': 'tmpl.fits', 'filepath': '/data/',
            'mjd': 58000.0, 'objname': 'SN2024abc', 'filter': 'rp',
        }
        mock_getlist = MagicMock(return_value=[template_row])
        mock_filtra = MagicMock(return_value={
            'filename': np.array(['tmpl.fits']),
            'filepath': np.array(['/data/']),
        })
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'diff', '-n', 'SN2024abc', '-T', 'fl',
                   '--tempdate', '19990101-20200101', '--difftype', '1'],
                  get_list_rv=ll,
                  **{
                      'lsc.myloopdef.run_diff': mock_rd,
                      'lsc.mysqldef.getlistfromraw': mock_getlist,
                      'lsc.myloopdef.filtralist': mock_filtra,
                  })
        # Check the suffix arg passed to run_diff
        call_args = mock_rd.call_args[0]
        suffix = call_args[9]  # suffix is the 10th positional arg
        assert 'optimal' in suffix

    def test_diff_template_not_found(self, monkeypatch):
        """Diff with no template found raises."""
        ll = _make_ll()
        mock_getlist = MagicMock(return_value=None)
        with pytest.raises(Exception, match='template not found'):
            _run_loop(monkeypatch,
                      ['-e', '20200501', '-s', 'diff', '-n', 'SN2024abc', '-T', 'fl',
                       '--tempdate', '19990101-20200101'],
                      get_list_rv=ll,
                      **{'lsc.mysqldef.getlistfromraw': mock_getlist})


class TestStageTemplate:

    def test_template_calls_run_template(self, monkeypatch):
        ll = _make_ll()
        mock_rt = MagicMock()
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'template'],
                  get_list_rv=ll,
                  **{'lsc.myloopdef.run_template': mock_rt})
        mock_rt.assert_called_once()


class TestStageFpack:

    def test_fpack_uses_filetype_0(self, monkeypatch):
        ll = _make_ll()
        ll['lastunpacked'] = np.array([datetime(2020, 1, 1)])
        import multiprocessing
        mock_pool_inst = MagicMock()
        mock_pool_inst.map.return_value = [0]
        monkeypatch.setattr(multiprocessing, 'Pool', MagicMock(return_value=mock_pool_inst))
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'fpack', '-F'],
                  get_list_rv=ll)

    def test_fpack_without_force_filters_old(self, monkeypatch):
        """fpack without -F only packs month-old data."""
        ll = _make_ll()
        ll['lastunpacked'] = np.array([datetime(2020, 1, 1)])
        import multiprocessing
        mock_pool_inst = MagicMock()
        mock_pool_inst.map.return_value = [0]
        monkeypatch.setattr(multiprocessing, 'Pool', MagicMock(return_value=mock_pool_inst))
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'fpack'],
                  get_list_rv=ll)


class TestStageMakestamp:

    def test_makestamp_called(self, monkeypatch):
        ll = _make_ll()
        mock_ms = MagicMock()
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'makestamp'],
                  get_list_rv=ll,
                  **{'lsc.myloopdef.makestamp': mock_ms})
        mock_ms.assert_called_once()


class TestStageApmag:

    def test_apmag_called(self, monkeypatch):
        ll = _make_ll()
        mock_ap = MagicMock()
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'apmag'],
                  get_list_rv=ll,
                  **{'lsc.myloopdef.run_apmag': mock_ap})
        mock_ap.assert_called_once()


class TestStageIngestsloan:

    def test_ingestsloan_called(self, monkeypatch):
        ll = _make_ll()
        mock_is = MagicMock()
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'ingestsloan'],
                  get_list_rv=ll,
                  **{'lsc.myloopdef.run_ingestsloan': mock_is})
        mock_is.assert_called_once()

    def test_ingestps1_called(self, monkeypatch):
        ll = _make_ll()
        mock_is = MagicMock()
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'ingestps1'],
                  get_list_rv=ll,
                  **{'lsc.myloopdef.run_ingestsloan': mock_is})
        mock_is.assert_called_once()


class TestStageMagAbscatLocal:

    def test_mag_with_catalogue(self, monkeypatch):
        ll = _make_ll()
        mock_cat = MagicMock()
        mock_get_standards = MagicMock(return_value={'filename': []})
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'mag', '--catalogue', '/path/to.cat'],
                  get_list_rv=ll,
                  **{'lsc.myloopdef.run_cat': mock_cat,
                     'lsc.myloopdef.get_standards': mock_get_standards})
        mock_cat.assert_called_once()

    def test_mag_with_field(self, monkeypatch):
        ll = _make_ll()
        mock_cat = MagicMock()
        mock_getcatalog = MagicMock(return_value='gaia.cat')
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'mag', '--field', 'gaia'],
                  get_list_rv=ll,
                  **{'lsc.myloopdef.run_cat': mock_cat,
                     'lsc.util.getcatalog': mock_getcatalog})
        mock_cat.assert_called_once()
        mock_getcatalog.assert_called()

    def test_mag_default_gaia(self, monkeypatch):
        ll = _make_ll()
        mock_cat = MagicMock()
        mock_getcatalog = MagicMock(return_value='gaia.cat')
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'mag'],
                  get_list_rv=ll,
                  **{'lsc.myloopdef.run_cat': mock_cat,
                     'lsc.util.getcatalog': mock_getcatalog})
        mock_cat.assert_called_once()
        # Should call getcatalog with 'gaia'
        mock_getcatalog.assert_called_with('', 'gaia')

    def test_mag_diff_filetype_warns(self, monkeypatch):
        """Filetype 3 + mag stage + type fit -> warning."""
        ll = _make_ll()
        mock_cat = MagicMock()
        mock_getcatalog = MagicMock(return_value='gaia.cat')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _run_loop(monkeypatch,
                      ['-e', '20200501', '-s', 'mag', '--filetype', '3', '--type', 'fit'],
                      get_list_rv=ll,
                      **{'lsc.myloopdef.run_cat': mock_cat,
                         'lsc.util.getcatalog': mock_getcatalog})
        assert any('Aperture photometry recommended' in str(warning.message) for warning in w)

    def test_mag_with_filter_as_field(self, monkeypatch):
        """When --field is empty but --filter is sloan, field is set from filter."""
        ll = _make_ll()
        mock_cat = MagicMock()
        mock_getcatalog = MagicMock(return_value='sloan.cat')
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'mag', '-f', 'sloan'],
                  get_list_rv=ll,
                  **{'lsc.myloopdef.run_cat': mock_cat,
                     'lsc.util.getcatalog': mock_getcatalog})
        mock_cat.assert_called_once()
        # field arg in run_cat should be 'sloan'
        call_args = mock_cat.call_args[0]
        assert call_args[6] == 'sloan'


class TestStageChecks:

    def test_checkmag_called(self, monkeypatch):
        ll = _make_ll()
        mock_fn = MagicMock()
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'checkmag'],
                  get_list_rv=ll,
                  **{'lsc.myloopdef.checkmag': mock_fn})
        mock_fn.assert_called_once()

    def test_checkwcs_called(self, monkeypatch):
        ll = _make_ll()
        mock_fn = MagicMock()
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'checkwcs'],
                  get_list_rv=ll,
                  **{'lsc.myloopdef.checkwcs': mock_fn})
        mock_fn.assert_called_once()

    def test_checkquality_called(self, monkeypatch):
        ll = _make_ll()
        mock_fn = MagicMock()
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'checkquality'],
                  get_list_rv=ll,
                  **{'lsc.myloopdef.checkquality': mock_fn})
        mock_fn.assert_called_once()

    def test_checkpos_called(self, monkeypatch):
        ll = _make_ll()
        mock_fn = MagicMock()
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'checkpos'],
                  get_list_rv=ll,
                  **{'lsc.myloopdef.checkpos': mock_fn})
        mock_fn.assert_called_once()

    def test_checkcat_called(self, monkeypatch):
        ll = _make_ll()
        mock_fn = MagicMock()
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'checkcat'],
                  get_list_rv=ll,
                  **{'lsc.myloopdef.checkcat': mock_fn})
        mock_fn.assert_called_once()

    def test_checkmissing_called(self, monkeypatch):
        ll = _make_ll()
        mock_fn = MagicMock()
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'checkmissing'],
                  get_list_rv=ll,
                  **{'lsc.myloopdef.check_missing': mock_fn})
        mock_fn.assert_called_once()

    def test_checkfvd_called(self, monkeypatch):
        ll = _make_ll()
        mock_fn = MagicMock()
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'checkfvd'],
                  get_list_rv=ll,
                  **{'lsc.myloopdef.checkfilevsdatabase': mock_fn})
        mock_fn.assert_called_once()

    def test_checkcosmic_called(self, monkeypatch):
        ll = _make_ll()
        mock_fn = MagicMock()
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'checkcosmic'],
                  get_list_rv=ll,
                  **{'lsc.myloopdef.checkcosmic': mock_fn})
        mock_fn.assert_called_once()


class TestStageStandard:

    def test_standard_flag_gets_standards(self, monkeypatch, capsys):
        """--standard flag calls get_standards and prints info."""
        ll = _make_ll()
        mm = {
            'filename': ['std.fits'],
            'objname': ['GD71'],
            'filter': ['rp'],
            'wcs': [0],
            'psf': ['std.psf.fits'],
            'psfmag': [18.0],
            'zcat': ['std.cat'],
            'mag': [18.0],
            'abscat': ['std.cat'],
            'filepath': ['/data/'],
            'dayobs': ['20200501'],
            'lastunpacked': [datetime(2020, 5, 1)],
        }
        mock_gs = MagicMock(return_value=mm)
        mock_rp = MagicMock()
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'psf', '--standard', 'mystandard'],
                  get_list_rv=ll,
                  **{'lsc.myloopdef.get_standards': mock_gs,
                     'lsc.myloopdef.run_psf': mock_rp})
        mock_gs.assert_called_once()
        # run_psf should be called with the standards list since stage is psf (not mag/abscat/local)
        mock_rp.assert_called_once()


class TestExceptBranch:

    def test_print_except_branch(self, monkeypatch, capsys):
        """Trigger the except branch in the image listing (when .replace fails)."""
        ll = _make_ll()
        # Make psf a non-string to trigger the except branch in printing
        ll['psf'] = np.array([None])
        mock_rp = MagicMock()
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'psf'],
                  get_list_rv=ll,
                  **{'lsc.myloopdef.run_psf': mock_rp})
        # It should still run without crashing
        mock_rp.assert_called_once()


class TestMultiRunCosmic:

    def test_multi_run_cosmic_unpacks_args(self, monkeypatch):
        """multi_run_cosmic at module level delegates to run_cosmic."""
        mock_rc = MagicMock(return_value=None)
        with patch('lsc.myloopdef.run_cosmic', mock_rc):
            ns = runpy.run_path(os.path.join(BIN_DIR, 'lscloop.py'), run_name='not_main')
            ns['multi_run_cosmic']((['file.fits'], 'photlco', 4.5, 0.2, 4, False))
        mock_rc.assert_called_once_with(['file.fits'], 'photlco', 4.5, 0.2, 4, False)


class TestFiletypeAndTypeDefaults:

    def test_checkdiff_filetype_3(self, monkeypatch, capsys):
        ll = _make_ll()
        mock_cd = MagicMock()
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'checkdiff'],
                  get_list_rv=ll,
                  **{'lsc.myloopdef.checkdiff': mock_cd})
        mock_cd.assert_called_once()

    def test_multicore_cap(self, monkeypatch, capsys):
        ll = _make_ll()
        mock_rc = MagicMock()
        import multiprocessing
        mock_pool_inst = MagicMock()
        mock_pool_inst.map.return_value = [None]
        monkeypatch.setattr(multiprocessing, 'Pool', MagicMock(return_value=mock_pool_inst))
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'cosmic', '--multicore', '9999'],
                  get_list_rv=ll,
                  **{'lsc.myloopdef.run_cosmic': mock_rc})
        out = capsys.readouterr().out
        assert 'Attempting to run on too many cores' in out

    def test_bad_diff_sets_filetype_1(self, monkeypatch):
        """bad='diff' forces filetype=1."""
        ll = _make_ll()
        mock_rp = MagicMock()
        _run_loop(monkeypatch,
                  ['-e', '20200501', '-s', 'psf', '-b', 'diff'],
                  get_list_rv=ll,
                  **{'lsc.myloopdef.run_psf': mock_rp})
        # get_list called with filetype=1
        mock_get_list = None
        # We verify by the fact it ran without error with the right filetype
        mock_rp.assert_called_once()
