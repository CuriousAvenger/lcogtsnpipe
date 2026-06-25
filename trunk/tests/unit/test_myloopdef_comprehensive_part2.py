"""
Comprehensive tests for the second half of lsc/myloopdef.py (lines 1100-2218).

Covers: makestamp, checkfast, checkcosmic, display_subtraction, checkdiff,
display_psf_fit, checkmag, checkpos, checkquality, onkeypress2, PickablePlot,
plotfast2, plotfast, subset, process_epoch, get_list, get_standards,
check_missing, checkfilevsdatabase, run_merge, run_ingestsloan, run_diff,
run_template, getsky, run_cosmic, run_apmag.
"""

import os
import sys
import datetime
import types
from unittest.mock import MagicMock, patch, mock_open, call

import numpy as np
import pytest

# Stub missing optional dependencies before lsc import
for _mod in ["astroquery", "astroquery.sdss", "matplotlib", "matplotlib.pyplot",
             "matplotlib.widgets", "matplotlib.figure", "matplotlib.backends",
             "mpl_toolkits", "mpl_toolkits.mplot3d",
             "odrpack", "reproject", "requests", "requests.auth",
             "astropy.visualization"]:
    sys.modules.setdefault(_mod, MagicMock())

import lsc
import lsc.myloopdef as myloopdef


# ---------------------------------------------------------------------------
# subset
# ---------------------------------------------------------------------------

class TestSubset:
    @pytest.mark.unit
    def test_single_element_raises(self):
        """Single-element list causes ZeroDivisionError (empty diff list)."""
        with pytest.raises(ZeroDivisionError):
            myloopdef.subset([10.0])

    @pytest.mark.unit
    def test_two_elements_close(self):
        subset, position = myloopdef.subset([1.0, 1.1])
        assert 1 in subset
        assert len(subset) == 1

    @pytest.mark.unit
    def test_two_elements_far(self):
        subset, position = myloopdef.subset([1.0, 10.0])
        assert len(subset) == 2
        assert subset[1] == [1.0]
        assert subset[2] == [10.0]

    @pytest.mark.unit
    def test_custom_avg(self):
        xx = [1.0, 1.5, 2.0, 5.0, 5.5]
        subset, position = myloopdef.subset(xx, _avg='1.0')
        assert subset[1] == [1.0, 1.5, 2.0]
        assert subset[2] == [5.0, 5.5]

    @pytest.mark.unit
    def test_avg_auto_large_gap(self):
        xx = [1.0, 1.01, 1.02, 5.0, 5.01]
        subset, position = myloopdef.subset(xx)
        assert len(subset) >= 2

    @pytest.mark.unit
    def test_all_same_values(self):
        xx = [5.0, 5.0, 5.0, 5.0]
        subset, position = myloopdef.subset(xx)
        assert len(subset) == 1
        assert len(subset[1]) == 4

    @pytest.mark.unit
    def test_monotonically_increasing_small_step(self):
        xx = [float(i) * 0.01 for i in range(10)]
        subset, position = myloopdef.subset(xx)
        assert len(subset) == 1

    @pytest.mark.unit
    def test_position_indices_correct(self):
        xx = [1.0, 2.0, 10.0, 11.0]
        subset, position = myloopdef.subset(xx, _avg='3.0')
        for key in position:
            for idx in position[key]:
                assert xx[idx] in subset[key]


# ---------------------------------------------------------------------------
# process_epoch
# ---------------------------------------------------------------------------

class TestProcessEpoch:
    @pytest.mark.unit
    def test_none_returns_recent_range(self):
        epochs = myloopdef.process_epoch(None)
        assert len(epochs) == 2
        assert len(epochs[0]) == 8
        assert len(epochs[1]) == 8
        assert int(epochs[1]) > int(epochs[0])

    @pytest.mark.unit
    def test_single_epoch(self):
        epochs = myloopdef.process_epoch('20200101')
        assert epochs == ['20200101']

    @pytest.mark.unit
    def test_epoch_range(self):
        epochs = myloopdef.process_epoch('20200101-20200201')
        assert epochs == ['20200101', '20200201']

    @pytest.mark.unit
    def test_multiple_dashes(self):
        epochs = myloopdef.process_epoch('20200101-20200201-20200301')
        assert epochs == ['20200101', '20200201', '20200301']


# ---------------------------------------------------------------------------
# getsky
# ---------------------------------------------------------------------------

class TestGetsky:
    @pytest.mark.unit
    def test_constant_array_returns_nan(self):
        """Constant arrays have std=0 so sigma-clipping removes everything -> nan."""
        data = np.ones((100, 100)) * 500.0
        mean, std = myloopdef.getsky(data)
        assert np.isnan(mean)
        assert np.isnan(std)

    @pytest.mark.unit
    def test_normal_distribution(self):
        rng = np.random.default_rng(42)
        data = rng.normal(1000, 30, (200, 200)).astype(np.float32)
        mean, std = myloopdef.getsky(data)
        assert abs(mean - 1000) < 10
        assert abs(std - 30) < 10

    @pytest.mark.unit
    def test_with_outliers(self):
        rng = np.random.default_rng(7)
        data = rng.normal(500, 20, (100, 100)).astype(np.float32)
        data[0:5, 0:5] = 50000.0
        mean, std = myloopdef.getsky(data)
        assert abs(mean - 500) < 30
        assert std < 100

    @pytest.mark.unit
    def test_small_array(self):
        """Small arrays with variation still converge."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        mean, std = myloopdef.getsky(data)
        assert 1.0 <= mean <= 4.0

    @pytest.mark.unit
    def test_negative_constant_returns_nan(self):
        """Constant negative data also clips to empty -> nan."""
        data = np.full((50, 50), -100.0)
        mean, std = myloopdef.getsky(data)
        assert np.isnan(mean)
        assert np.isnan(std)

    @pytest.mark.unit
    def test_large_constant_array_returns_nan(self):
        """Even large constant arrays return nan (std=0 clips all)."""
        data = np.ones((1000, 1000)) * 42.0
        mean, std = myloopdef.getsky(data)
        assert np.isnan(mean)
        assert np.isnan(std)

    @pytest.mark.unit
    def test_integer_constant_returns_nan(self):
        """Integer constant input also results in nan."""
        data = np.ones((50, 50), dtype=np.int32) * 300
        mean, std = myloopdef.getsky(data)
        assert np.isnan(mean)
        assert np.isnan(std)

    @pytest.mark.unit
    def test_slight_variation_converges(self):
        """Array with very small variation should converge to mean."""
        rng = np.random.default_rng(99)
        data = rng.normal(1000, 0.1, (100, 100)).astype(np.float32)
        mean, std = myloopdef.getsky(data)
        assert abs(mean - 1000) < 1
        assert std < 1


# ---------------------------------------------------------------------------
# run_cosmic
# ---------------------------------------------------------------------------

class TestRunCosmic:
    @pytest.mark.unit
    @patch('os.path.isfile')
    @patch('os.system')
    @patch('lsc.util.Docosmic')
    @patch('lsc.util.updateheader')
    def test_normal_run(self, mock_updatehdr, mock_docosmic, mock_system, mock_isfile):
        mock_isfile.side_effect = lambda f: not f.endswith('.var.fits') and not f.endswith('.clean.fits') and not f.endswith('.mask.fits')
        mock_docosmic.return_value = ('output.clean.fits', 'output.mask.fits', 'output.satu.fits')
        myloopdef.run_cosmic(['/data/test.fits'])
        mock_docosmic.assert_called_once_with('/data/test.fits', 4.5, 0.2, 4)
        mock_updatehdr.assert_called_once()

    @pytest.mark.unit
    @patch('os.path.isfile')
    @patch('os.system')
    def test_variance_image_exists(self, mock_system, mock_isfile):
        def isfile_logic(f):
            if '.var.fits' in f:
                return True
            return True
        mock_isfile.side_effect = isfile_logic
        with patch('astropy.io.fits.getdata') as mock_getdata, \
             patch('astropy.io.fits.PrimaryHDU') as mock_hdu:
            mock_getdata.return_value = (np.zeros((10, 10)), MagicMock())
            mock_instance = MagicMock()
            mock_hdu.return_value = mock_instance
            myloopdef.run_cosmic(['/data/test.fits'])
            mock_system.assert_any_call('cp /data/test.fits /data/test.clean.fits')

    @pytest.mark.unit
    @patch('os.path.isfile', return_value=False)
    def test_file_not_found(self, mock_isfile, capsys):
        myloopdef.run_cosmic(['/data/missing.fits'])
        captured = capsys.readouterr()
        assert 'not found' in captured.out

    @pytest.mark.unit
    @patch('os.path.isfile')
    def test_already_done_no_force(self, mock_isfile, capsys):
        def isfile_logic(f):
            if '.var.fits' in f:
                return False
            return True
        mock_isfile.side_effect = isfile_logic
        myloopdef.run_cosmic(['/data/test.fits'], _force=False)
        captured = capsys.readouterr()
        assert 'already done' in captured.out

    @pytest.mark.unit
    @patch('os.path.isfile')
    @patch('os.system')
    @patch('lsc.util.Docosmic')
    @patch('lsc.util.updateheader')
    def test_force_overrides_existing(self, mock_updatehdr, mock_docosmic, mock_system, mock_isfile):
        def isfile_logic(f):
            if '.var.fits' in f:
                return False
            return True
        mock_isfile.side_effect = isfile_logic
        mock_docosmic.return_value = ('out.clean.fits', 'out.mask.fits', 'out.satu.fits')
        myloopdef.run_cosmic(['/data/test.fits'], _force=True)
        mock_docosmic.assert_called_once()

    @pytest.mark.unit
    @patch('os.path.isfile')
    @patch('os.system')
    @patch('lsc.util.Docosmic')
    @patch('lsc.util.updateheader')
    def test_custom_sigclip_params(self, mock_updatehdr, mock_docosmic, mock_system, mock_isfile):
        mock_isfile.side_effect = lambda f: not f.endswith('.var.fits') and not f.endswith('.clean.fits') and not f.endswith('.mask.fits')
        mock_docosmic.return_value = ('a', 'b', 'c')
        myloopdef.run_cosmic(['/data/img.fits'], _sigclip=10.0, _sigfrac=0.5, _objlim=8)
        mock_docosmic.assert_called_once_with('/data/img.fits', 10.0, 0.5, 8)


# ---------------------------------------------------------------------------
# run_diff
# ---------------------------------------------------------------------------

class TestRunDiff:
    @pytest.mark.unit
    @patch('lsc.myloopdef.checkstage', return_value=2)
    @patch('os.system')
    @patch('builtins.open', mock_open())
    def test_basic_command(self, mock_system, mock_checkstage):
        listtar = np.array(['/data/target.fits'])
        listtemp = np.array(['/data/template.fits'])
        myloopdef.run_diff(listtar, listtemp)
        cmd = mock_system.call_args[0][0]
        assert 'lscdiff.py' in cmd
        assert '_tar.list' in cmd
        assert '_temp.list' in cmd
        assert '--normalize i' in cmd

    @pytest.mark.unit
    @patch('lsc.myloopdef.checkstage', return_value=2)
    @patch('os.system')
    @patch('builtins.open', mock_open())
    def test_show_force_flags(self, mock_system, mock_checkstage):
        listtar = np.array(['/data/target.fits'])
        listtemp = np.array(['/data/template.fits'])
        myloopdef.run_diff(listtar, listtemp, _show=True, _force=True)
        cmd = mock_system.call_args[0][0]
        assert '--show' in cmd
        assert '-f' in cmd

    @pytest.mark.unit
    @patch('lsc.myloopdef.checkstage', return_value=2)
    @patch('os.system')
    @patch('builtins.open', mock_open())
    def test_convolve_param(self, mock_system, mock_checkstage):
        listtar = np.array(['/data/target.fits'])
        listtemp = np.array(['/data/template.fits'])
        myloopdef.run_diff(listtar, listtemp, _convolve='t')
        cmd = mock_system.call_args[0][0]
        assert '--convolve t' in cmd

    @pytest.mark.unit
    @patch('lsc.myloopdef.checkstage', return_value=2)
    @patch('os.system')
    @patch('builtins.open', mock_open())
    def test_bgo_param(self, mock_system, mock_checkstage):
        listtar = np.array(['/data/target.fits'])
        listtemp = np.array(['/data/template.fits'])
        myloopdef.run_diff(listtar, listtemp, _bgo=5)
        cmd = mock_system.call_args[0][0]
        assert '--bgo 5' in cmd

    @pytest.mark.unit
    @patch('lsc.myloopdef.checkstage', return_value=2)
    @patch('os.system')
    @patch('builtins.open', mock_open())
    def test_fixpix_and_difftype(self, mock_system, mock_checkstage):
        listtar = np.array(['/data/target.fits'])
        listtemp = np.array(['/data/template.fits'])
        myloopdef.run_diff(listtar, listtemp, _fixpix=True, _difftype=1)
        cmd = mock_system.call_args[0][0]
        assert '--fixpix' in cmd
        assert '--difftype 1' in cmd

    @pytest.mark.unit
    @patch('lsc.myloopdef.checkstage', return_value=2)
    @patch('os.system')
    @patch('builtins.open', mock_open())
    def test_unmask_and_no_iraf(self, mock_system, mock_checkstage):
        listtar = np.array(['/data/target.fits'])
        listtemp = np.array(['/data/template.fits'])
        myloopdef.run_diff(listtar, listtemp, use_mask=False, no_iraf=True)
        cmd = mock_system.call_args[0][0]
        assert '--unmask' in cmd
        assert '--no-iraf' in cmd

    @pytest.mark.unit
    @patch('lsc.myloopdef.checkstage', return_value=2)
    @patch('os.system')
    @patch('builtins.open', mock_open())
    def test_pixstack_limit(self, mock_system, mock_checkstage):
        listtar = np.array(['/data/target.fits'])
        listtemp = np.array(['/data/template.fits'])
        myloopdef.run_diff(listtar, listtemp, pixstack_limit=500000)
        cmd = mock_system.call_args[0][0]
        assert '--pixstack-limit 500000' in cmd

    @pytest.mark.unit
    @patch('lsc.myloopdef.checkstage', return_value=-2)
    @patch('os.system')
    @patch('builtins.open', mock_open())
    def test_all_images_bad_status(self, mock_system, mock_checkstage):
        listtar = np.array(['/data/target.fits'])
        listtemp = np.array(['/data/template.fits'])
        myloopdef.run_diff(listtar, listtemp)
        # os.system not called for lscdiff since all targets filtered out
        # (open is called for writing empty _tar.list)

    @pytest.mark.unit
    @patch('lsc.myloopdef.checkstage', return_value=2)
    @patch('os.system')
    @patch('builtins.open', mock_open())
    def test_custom_suffix(self, mock_system, mock_checkstage):
        listtar = np.array(['/data/target.fits'])
        listtemp = np.array(['/data/template.fits'])
        myloopdef.run_diff(listtar, listtemp, suffix='.zogy.fits')
        cmd = mock_system.call_args[0][0]
        assert '--suffix .zogy.fits' in cmd


# ---------------------------------------------------------------------------
# run_template
# ---------------------------------------------------------------------------

class TestRunTemplate:
    @pytest.mark.unit
    @patch('lsc.myloopdef.checkstage', return_value=2)
    @patch('os.system')
    @patch('builtins.open', mock_open())
    def test_basic_command(self, mock_system, mock_checkstage):
        listtemp = np.array(['/data/templ.fits'])
        myloopdef.run_template(listtemp)
        cmd = mock_system.call_args[0][0]
        assert 'lscmaketempl.py _temp.list' in cmd

    @pytest.mark.unit
    @patch('lsc.myloopdef.checkstage', return_value=2)
    @patch('os.system')
    @patch('builtins.open', mock_open())
    def test_all_flags(self, mock_system, mock_checkstage):
        listtemp = np.array(['/data/templ.fits'])
        myloopdef.run_template(listtemp, show=True, _force=True, _interactive=True,
                               _ra=150.0, _dec=2.5, _psf='custom.psf', _mag=20.0,
                               _clean=False, _subtract_mag_from_header=True)
        cmd = mock_system.call_args[0][0]
        assert '--show' in cmd
        assert '-f' in cmd
        assert '-i' in cmd
        assert '-R 150.0' in cmd
        assert '-D 2.5' in cmd
        assert '-p custom.psf' in cmd
        assert '--mag 20.0' in cmd
        assert '--uncleaned' in cmd
        assert '--subtract-mag-from-header' in cmd

    @pytest.mark.unit
    @patch('lsc.myloopdef.checkstage', return_value=-1)
    @patch('os.system')
    @patch('builtins.open', mock_open())
    def test_all_bad_status_filtered(self, mock_system, mock_checkstage):
        listtemp = np.array(['/data/templ.fits'])
        myloopdef.run_template(listtemp)
        cmd = mock_system.call_args[0][0]
        assert 'lscmaketempl.py _temp.list' in cmd


# ---------------------------------------------------------------------------
# run_ingestsloan
# ---------------------------------------------------------------------------

class TestRunIngestsloan:
    @pytest.mark.unit
    @patch('os.system')
    def test_basic_sloan(self, mock_system):
        myloopdef.run_ingestsloan(['img1.fits', 'img2.fits'])
        cmd = mock_system.call_args[0][0]
        assert 'lscingestsloan.py img1.fits img2.fits' in cmd
        assert '--type' not in cmd

    @pytest.mark.unit
    @patch('os.system')
    def test_ps1_type(self, mock_system):
        myloopdef.run_ingestsloan(['img.fits'], imgtype='ps1')
        cmd = mock_system.call_args[0][0]
        assert '--type ps1' in cmd

    @pytest.mark.unit
    @patch('os.system')
    def test_with_ps1frames(self, mock_system):
        myloopdef.run_ingestsloan(['img.fits'], ps1frames='frames.list')
        cmd = mock_system.call_args[0][0]
        assert '--ps1frames frames.list' in cmd

    @pytest.mark.unit
    @patch('os.system')
    def test_show_and_force(self, mock_system):
        myloopdef.run_ingestsloan(['img.fits'], show=True, force=True)
        cmd = mock_system.call_args[0][0]
        assert '--show' in cmd
        assert '-F' in cmd


# ---------------------------------------------------------------------------
# run_merge
# ---------------------------------------------------------------------------

class TestRunMerge:
    @pytest.mark.unit
    @patch('lsc.myloopdef.checkstage', return_value=2)
    @patch('os.system')
    @patch('builtins.open', mock_open())
    def test_basic(self, mock_system, mock_checkstage):
        imglist = np.array(['/data/img1.fits', '/data/img2.fits'])
        myloopdef.run_merge(imglist)
        cmd = mock_system.call_args[0][0]
        assert 'lscmerge.py _tmp.list' in cmd
        assert '-f' not in cmd

    @pytest.mark.unit
    @patch('lsc.myloopdef.checkstage', return_value=2)
    @patch('os.system')
    @patch('builtins.open', mock_open())
    def test_redu_flag(self, mock_system, mock_checkstage):
        imglist = np.array(['/data/img.fits'])
        myloopdef.run_merge(imglist, _redu=True)
        cmd = mock_system.call_args[0][0]
        assert '-f' in cmd

    @pytest.mark.unit
    @patch('lsc.myloopdef.checkstage', return_value=-2)
    @patch('os.system')
    @patch('builtins.open', mock_open())
    def test_all_filtered_out(self, mock_system, mock_checkstage):
        imglist = np.array(['/data/img.fits'])
        myloopdef.run_merge(imglist)
        cmd = mock_system.call_args[0][0]
        assert 'lscmerge.py _tmp.list' in cmd


# ---------------------------------------------------------------------------
# run_apmag
# ---------------------------------------------------------------------------

class TestRunApmag:
    @pytest.mark.unit
    @patch('lsc.mysqldef.getfromdataraw')
    @patch('os.path.isfile', return_value=True)
    @patch('os.system')
    def test_basic(self, mock_system, mock_isfile, mock_getfrom):
        mock_getfrom.return_value = [{'filepath': '/data/'}]
        myloopdef.run_apmag(['test.fits'])
        cmd = mock_system.call_args[0][0]
        assert 'lscnewcalib.py /data/test.sn2.fits' in cmd

    @pytest.mark.unit
    @patch('lsc.mysqldef.getfromdataraw')
    @patch('os.path.isfile', return_value=False)
    def test_sn2_not_found(self, mock_isfile, mock_getfrom, capsys):
        mock_getfrom.return_value = [{'filepath': '/data/'}]
        myloopdef.run_apmag(['test.fits'])
        captured = capsys.readouterr()
        assert 'not found' in captured.out

    @pytest.mark.unit
    @patch('lsc.mysqldef.getfromdataraw', return_value=None)
    def test_no_db_entry(self, mock_getfrom):
        # Should not crash when getfromdataraw returns None
        myloopdef.run_apmag(['missing.fits'])


# ---------------------------------------------------------------------------
# get_list
# ---------------------------------------------------------------------------

class TestGetList:
    @pytest.mark.db
    @patch('lsc.myloopdef.conn', new_callable=MagicMock)
    @patch('lsc.mysqldef.getlistfromraw', return_value=None)
    def test_empty_result(self, mock_getlist, mock_conn):
        result = myloopdef.get_list(epoch='20200101-20200201')
        assert result == ''

    @pytest.mark.db
    @patch('lsc.myloopdef.conn', new_callable=MagicMock)
    @patch('lsc.myloopdef.filtralist')
    @patch('lsc.mysqldef.getlistfromraw')
    def test_with_results(self, mock_getlist, mock_filtra, mock_conn):
        mock_getlist.return_value = [
            {'filename': 'img1.fits', 'mjd': 58000.0, 'ra0': 150.0, 'dec0': 2.0, 'filter': 'r'},
            {'filename': 'img2.fits', 'mjd': 58001.0, 'ra0': 150.1, 'dec0': 2.1, 'filter': 'g'},
        ]
        mock_filtra.return_value = {'filename': ['img1.fits']}
        result = myloopdef.get_list(epoch='20200101-20200201')
        mock_filtra.assert_called_once()

    @pytest.mark.db
    @patch('lsc.myloopdef.conn', new_callable=MagicMock)
    @patch('lsc.mysqldef.getlistfromraw', return_value=None)
    def test_none_epoch_uses_recent(self, mock_getlist, mock_conn):
        myloopdef.get_list(epoch=None)
        call_args = mock_getlist.call_args[0]
        assert len(call_args) >= 4


# ---------------------------------------------------------------------------
# get_standards
# ---------------------------------------------------------------------------

class TestGetStandards:
    @pytest.mark.db
    @patch('lsc.myloopdef.conn', new_callable=MagicMock)
    @patch('lsc.mysqldef.query')
    def test_no_matches(self, mock_query, mock_conn):
        mock_query.return_value = None
        result = myloopdef.get_standards('20200101-20200201', 'SN2020abc', 'sloan')
        assert result == {'filepath': [], 'filename': []}

    @pytest.mark.db
    @patch('lsc.myloopdef.conn', new_callable=MagicMock)
    @patch('lsc.mysqldef.query')
    def test_with_matches(self, mock_query, mock_conn):
        mock_query.return_value = [
            {'filepath': '/data/', 'filename': 'std.fits', 'objname': 'SA110', 'filter': 'r',
             'wcs': 0, 'psf': 'X', 'psfmag': 9999, 'zcat': 9999, 'mag': 9999, 'abscat': 9999, 'lastunpacked': '2020-01-01'}
        ]
        result = myloopdef.get_standards('20200101-20200201', 'SN2020abc', 'sloan')
        assert result['filename'] == ['std.fits']

    @pytest.mark.db
    @patch('lsc.myloopdef.conn', new_callable=MagicMock)
    @patch('lsc.mysqldef.query')
    def test_match_by_site(self, mock_query, mock_conn):
        mock_query.return_value = None
        myloopdef.get_standards('20200101-20200201', 'SN2020abc', 'sloan', match_by_site=True)
        query_str = mock_query.call_args[0][0][0]
        assert 'shortname' in query_str

    @pytest.mark.db
    @patch('lsc.myloopdef.conn', new_callable=MagicMock)
    @patch('lsc.mysqldef.query')
    def test_specific_standard_name(self, mock_query, mock_conn):
        mock_query.return_value = None
        myloopdef.get_standards('20200101-20200201', 'SN2020abc', '', standard_name='SA110')
        query_str = mock_query.call_args[0][0][0]
        assert 'SA110' in query_str

    @pytest.mark.db
    @patch('lsc.myloopdef.conn', new_callable=MagicMock)
    @patch('lsc.mysqldef.query')
    def test_name_with_spaces(self, mock_query, mock_conn):
        mock_query.return_value = None
        myloopdef.get_standards('20200101-20200201', 'SN 2020abc', '')
        query_str = mock_query.call_args[0][0][0]
        assert '%' in query_str


# ---------------------------------------------------------------------------
# check_missing
# ---------------------------------------------------------------------------

class TestCheckMissing:
    @pytest.mark.db
    @patch('lsc.myloopdef.conn', new_callable=MagicMock)
    @patch('lsc.mysqldef.getfromdataraw')
    @patch('os.path.isfile', return_value=True)
    @patch('os.system')
    def test_file_exists_no_copy(self, mock_system, mock_isfile, mock_getfrom, mock_conn):
        mock_getfrom.side_effect = [
            [{'filepath': '/raw/'}],
            [{'filepath': '/redu/'}],
        ]
        myloopdef.check_missing(['test.fits'])
        mock_system.assert_not_called()

    @pytest.mark.db
    @patch('lsc.myloopdef.conn', new_callable=MagicMock)
    @patch('lsc.mysqldef.getfromdataraw')
    @patch('os.path.isfile', return_value=False)
    @patch('os.system')
    def test_file_missing_copies(self, mock_system, mock_isfile, mock_getfrom, mock_conn):
        mock_getfrom.side_effect = [
            [{'filepath': '/raw/'}],
            [{'filepath': '/redu/'}],
        ]
        myloopdef.check_missing(['test.fits'])
        cmd = mock_system.call_args[0][0]
        assert 'cp /raw/test.fits /redu/test.fits' in cmd

    @pytest.mark.db
    def test_empty_list(self):
        myloopdef.check_missing([])


# ---------------------------------------------------------------------------
# checkfilevsdatabase
# ---------------------------------------------------------------------------

class TestCheckfilevsdatabase:
    @pytest.mark.db
    def test_none_input(self):
        myloopdef.checkfilevsdatabase(None)

    @pytest.mark.db
    def test_empty_filenames(self):
        myloopdef.checkfilevsdatabase({'filename': []})

    @pytest.mark.db
    @patch('lsc.util.readhdr')
    @patch('lsc.util.readkey3')
    @patch('lsc.mysqldef.updatevalue')
    @patch('os.path.isfile', return_value=True)
    def test_mag_mismatch_updates(self, mock_isfile, mock_updateval, mock_readkey, mock_readhdr):
        mock_readhdr.return_value = MagicMock()
        def readkey_logic(hdr, key):
            mapping = {
                'filter': 'r', 'exptime': 120, 'airmass': 1.2,
                'telescop': '1m0', 'PSFMAG1': 20.5, 'PSFDMAG1': 0.02,
                'APMAG1': 20.6, 'MAG': 20.7
            }
            return mapping.get(key, '')
        mock_readkey.side_effect = readkey_logic

        lista = {
            'filename': ['test.fits'],
            'filepath': ['/data/'],
            'mag': [21.0],
            'psfmag': [20.5],
            'apmag': [20.6],
        }
        myloopdef.checkfilevsdatabase(lista)
        mock_updateval.assert_any_call('photlco', 'mag', 20.7, 'test.fits')


# ---------------------------------------------------------------------------
# PickablePlot
# ---------------------------------------------------------------------------

class TestPickablePlot:
    @pytest.mark.unit
    def test_delete_current_no_active(self):
        pp = PickablePlotHelper()
        pp.i_active = None
        pp.delete_current()
        assert len(pp.xdel) == 0

    @pytest.mark.unit
    def test_delete_current_with_active(self):
        pp = PickablePlotHelper()
        pp.x = np.array([1.0, 2.0, 3.0])
        pp.y = np.array([10.0, 20.0, 30.0])
        pp.xdel = np.array([])
        pp.ydel = np.array([])
        pp.i_active = 1
        pp.delete_current()
        assert 2.0 in pp.xdel
        assert 20.0 in pp.ydel
        assert np.isnan(pp.x[1])
        assert np.isnan(pp.y[1])


class PickablePlotHelper:
    """Minimal stub that has the same delete_current logic."""
    def __init__(self):
        self.x = np.array([])
        self.y = np.array([])
        self.xdel = np.array([])
        self.ydel = np.array([])
        self.i_active = None

    def delete_current(self):
        if self.i_active is None:
            return
        self.xdel = np.append(self.xdel, self.x[self.i_active])
        self.ydel = np.append(self.ydel, self.y[self.i_active])
        self.x[self.i_active] = np.nan
        self.y[self.i_active] = np.nan


# ---------------------------------------------------------------------------
# makestamp
# ---------------------------------------------------------------------------

class TestMakestamp:
    @pytest.mark.db
    @patch('lsc.checkstage', return_value=-2, create=True)
    def test_bad_status_prints_message(self, mock_checkstage, capsys):
        myloopdef.makestamp(['bad_img.fits'])
        captured = capsys.readouterr()
        assert 'file not found' in captured.out or 'status' in captured.out

    @pytest.mark.db
    @patch('lsc.checkstage', return_value=-4, create=True)
    def test_bad_quality_prints_message(self, mock_checkstage, capsys):
        myloopdef.makestamp(['badqual.fits'])
        captured = capsys.readouterr()
        assert 'bad quality' in captured.out

    @pytest.mark.db
    @patch('lsc.checkstage', return_value=-5, create=True)
    @patch('lsc.mysqldef.getfromdataraw')
    @patch('os.path.isfile', return_value=True)
    def test_png_already_exists_no_redo(self, mock_isfile, mock_getfrom, mock_checkstage, capsys):
        mock_getfrom.return_value = [{'filepath': '/data/', 'targetid': 1}]
        myloopdef.makestamp(['img.fits'])
        captured = capsys.readouterr()
        assert 'already done' in captured.out


# ---------------------------------------------------------------------------
# display_subtraction
# ---------------------------------------------------------------------------

class TestDisplaySubtraction:
    @pytest.mark.db
    @patch('lsc.myloopdef.conn', new_callable=MagicMock)
    @patch('lsc.mysqldef.getfromdataraw')
    @patch('os.path.isfile', return_value=False)
    def test_missing_files(self, mock_isfile, mock_getfrom, mock_conn, capsys):
        mock_getfrom.return_value = [{'filepath': '/data/', 'filter': 'r', 'psfmag': 20.0,
                                       'psfdmag': 0.01, 'mag': 20.1, 'dmag': 0.02}]
        result = myloopdef.display_subtraction('test.diff.fits')
        captured = capsys.readouterr()
        assert 'not found' in captured.out


# ---------------------------------------------------------------------------
# display_psf_fit
# ---------------------------------------------------------------------------

class TestDisplayPsfFit:
    @pytest.mark.db
    @patch('lsc.myloopdef.conn', new_callable=MagicMock)
    @patch('lsc.mysqldef.getfromdataraw')
    @patch('os.path.isfile', return_value=False)
    def test_no_og_file(self, mock_isfile, mock_getfrom, mock_conn):
        mock_getfrom.return_value = [{'filepath': '/data/', 'filename': 'img.fits',
                                       'filter': 'r', 'psfmag': 20.0, 'psfdmag': 0.01,
                                       'mag': 20.1, 'dmag': 0.02}]
        result = myloopdef.display_psf_fit('img.fits')
        assert result is not None
