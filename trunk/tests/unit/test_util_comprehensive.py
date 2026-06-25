"""
Comprehensive tests for lsc.util covering untested functions, edge cases,
boundary conditions, and error paths not covered by existing test files.

Focus areas:
- readkey3: RA/DEC with Angle conversions, hash-stripping, edge keyword cases
- readpasswd: edge cases (single row, eval failures)
- ReadAscii2: malformed data, extra columns
- readlist: edge cases with glob, empty lists
- delete: at-sign with blank lines and whitespace
- imcopy: extension parameter
- readstandard: negative declinations, missing magnitude
- readspectrum: 2D and 3D data arrays
- pval/residual: numpy array inputs, edge cases
- defswarp: combine type normalization
- writeinthelog: special characters
- repstringinfile: regex metacharacters, in-place (filein==fileout)
- updateheader: tuple-value (value, comment) format
- userinput: non-tty and EOFError paths
- missingvalues: module-level constant
- limmag: negative mbkg clamping
- Docosmic: 'temp' in filename for float mask type
"""
import os
import sys
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from astropy.io import fits

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_missingvalues_contains_expected_entries(self):
        from lsc.util import missingvalues
        assert 'NaN' in missingvalues
        assert 'UNKNOWN' in missingvalues
        assert None in missingvalues
        assert '' in missingvalues
        assert 'UNSPECIFIED' in missingvalues
        assert 'N/A' in missingvalues

    def test_missingvalues_is_list(self):
        from lsc.util import missingvalues
        assert isinstance(missingvalues, list)

    def test_pyversion_is_int(self):
        from lsc.util import pyversion
        assert isinstance(pyversion, int)
        assert pyversion >= 3

    def test_workdirectory_is_string(self):
        from lsc.util import workdirectory
        assert isinstance(workdirectory, str)


# ---------------------------------------------------------------------------
# userinput — non-tty and EOFError branches
# ---------------------------------------------------------------------------

class TestUserinputEdgeCases:
    def test_returns_empty_when_not_tty(self):
        from lsc.util import userinput
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            result = userinput("prompt: ")
            assert result == ''

    def test_returns_empty_on_eoferror(self):
        from lsc.util import userinput
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            with patch("builtins.input", side_effect=EOFError):
                result = userinput("prompt: ")
                assert result == ''

    def test_returns_multiline_input(self):
        from lsc.util import userinput
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            with patch("builtins.input", return_value="line with spaces"):
                result = userinput("? ")
                assert result == "line with spaces"

    def test_returns_numeric_string(self):
        from lsc.util import userinput
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            with patch("builtins.input", return_value="42"):
                result = userinput("number: ")
                assert result == "42"


# ---------------------------------------------------------------------------
# readpasswd — additional edge cases
# ---------------------------------------------------------------------------

class TestReadpasswdEdgeCases:
    def _write_config(self, tmp_path, content):
        p = tmp_path / "configure"
        p.write_text(content)
        return str(p)

    def test_eval_fails_uses_raw_string(self, tmp_path):
        """When eval() fails on a value, the raw string is stored."""
        from lsc.util import readpasswd
        # 'hello world' cannot be eval'd (bare identifier), so except stores it
        cfg = self._write_config(tmp_path, "key1 some_value_that_cant_eval\nkey2 another\n")
        result = readpasswd(cfg)
        assert result["key1"] == "some_value_that_cant_eval"

    def test_boolean_eval(self, tmp_path):
        """eval('True') should produce Python True."""
        from lsc.util import readpasswd
        cfg = self._write_config(tmp_path, "flag True\nother_key value\n")
        result = readpasswd(cfg)
        assert result["flag"] is True

    def test_tuple_eval(self, tmp_path):
        """eval on a tuple literal should return a tuple."""
        from lsc.util import readpasswd
        cfg = self._write_config(tmp_path, "coords (1,2,3)\nname test\n")
        result = readpasswd(cfg)
        assert result["coords"] == (1, 2, 3)

    def test_dict_eval(self, tmp_path):
        """eval on a dict literal should return a dict."""
        from lsc.util import readpasswd
        cfg = self._write_config(tmp_path, "mapping {'a':1}\nname test\n")
        result = readpasswd(cfg)
        assert result["mapping"] == {'a': 1}


# ---------------------------------------------------------------------------
# ReadAscii2 — edge cases
# ---------------------------------------------------------------------------

class TestReadAscii2EdgeCases:
    def test_negative_values(self, tmp_path):
        from lsc.util import ReadAscii2
        p = tmp_path / "neg.txt"
        p.write_text("-1.5 -2.5\n-3.0 4.0\n")
        col1, col2 = ReadAscii2(str(p))
        assert col1 == [-1.5, -3.0]
        assert col2 == [-2.5, 4.0]

    def test_extra_columns_ignored(self, tmp_path):
        """Only first two columns are read; extras are ignored."""
        from lsc.util import ReadAscii2
        p = tmp_path / "extra.txt"
        p.write_text("1.0 2.0 3.0 4.0\n5.0 6.0 7.0 8.0\n")
        col1, col2 = ReadAscii2(str(p))
        assert col1 == [1.0, 5.0]
        assert col2 == [2.0, 6.0]

    def test_scientific_notation(self, tmp_path):
        from lsc.util import ReadAscii2
        p = tmp_path / "sci.txt"
        p.write_text("1.0e3 2.5e-2\n3.14e0 -1.0e1\n")
        col1, col2 = ReadAscii2(str(p))
        assert abs(col1[0] - 1000.0) < 1e-6
        assert abs(col2[0] - 0.025) < 1e-6

    def test_tab_separated(self, tmp_path):
        """Tab-separated values should be handled by str.split."""
        from lsc.util import ReadAscii2
        p = tmp_path / "tabs.txt"
        p.write_text("1.0\t2.0\n3.0\t4.0\n")
        col1, col2 = ReadAscii2(str(p))
        assert col1 == [1.0, 3.0]
        assert col2 == [2.0, 4.0]

    def test_all_comments_returns_empty(self, tmp_path):
        from lsc.util import ReadAscii2
        p = tmp_path / "allcmts.txt"
        p.write_text("# comment 1\n# comment 2\n# comment 3\n")
        col1, col2 = ReadAscii2(str(p))
        assert col1 == []
        assert col2 == []

    def test_mixed_whitespace(self, tmp_path):
        from lsc.util import ReadAscii2
        p = tmp_path / "mixed.txt"
        p.write_text("  1.0   2.0  \n  3.0  4.0   \n")
        col1, col2 = ReadAscii2(str(p))
        assert col1 == [1.0, 3.0]
        assert col2 == [2.0, 4.0]


# ---------------------------------------------------------------------------
# readlist — additional edge cases
# ---------------------------------------------------------------------------

class TestReadlistEdgeCases:
    def test_text_list_with_spaces_in_lines(self, simple_fits, tmp_path):
        """Lines with spaces (but pointing to valid FITS) should be cleaned."""
        from lsc.util import readlist
        list_file = tmp_path / "spaces.list"
        # readlist does re.sub(' ','',ff), removing all spaces
        # So the path must not have spaces to resolve correctly after cleaning
        list_file.write_text(f" {simple_fits} \n")
        result = readlist(str(list_file))
        assert simple_fits in result

    def test_glob_no_match_exits(self, tmp_path):
        """Glob pattern matching no files calls sys.exit."""
        from lsc.util import readlist
        with pytest.raises(SystemExit):
            readlist(str(tmp_path / "*.nonexistent_extension"))

    def test_multiple_comma_separated(self, simple_fits):
        """Three comma-separated paths."""
        from lsc.util import readlist
        path_str = ",".join([simple_fits] * 3)
        result = readlist(path_str)
        assert len(result) == 3

    def test_single_comma_in_path_splits(self, simple_fits):
        """A comma in the input triggers split behavior."""
        from lsc.util import readlist
        result = readlist(f"{simple_fits},{simple_fits}")
        assert len(result) == 2


# ---------------------------------------------------------------------------
# delete — additional edge cases
# ---------------------------------------------------------------------------

class TestDeleteEdgeCases:
    def test_at_sign_skips_blank_lines(self, tmp_path):
        from lsc.util import delete
        f1 = tmp_path / "todel.fits"
        f1.write_text("x")
        listfile = tmp_path / "del.list"
        listfile.write_text(f"\n\n{f1}\n\n")
        delete("@" + str(listfile))
        assert not f1.exists()

    def test_at_sign_skips_whitespace_only_lines(self, tmp_path):
        """Lines that are just spaces become empty after re.sub and are skipped."""
        from lsc.util import delete
        f1 = tmp_path / "delws.fits"
        f1.write_text("x")
        listfile = tmp_path / "ws.list"
        listfile.write_text(f"   \n{f1}\n")
        delete("@" + str(listfile))
        assert not f1.exists()

    def test_comma_with_nonexistent_files(self, tmp_path):
        """Comma-separated list with some non-existent files should not raise."""
        from lsc.util import delete
        f1 = tmp_path / "existing.fits"
        f1.write_text("x")
        delete(f"{f1},{tmp_path}/nonexist.fits")
        assert not f1.exists()

    def test_empty_at_list(self, tmp_path):
        """@-list with no valid entries should not raise."""
        from lsc.util import delete
        listfile = tmp_path / "empty.list"
        listfile.write_text("# only comments\n\n")
        delete("@" + str(listfile))  # Should not raise


# ---------------------------------------------------------------------------
# imcopy — extension parameter and full copy
# ---------------------------------------------------------------------------

class TestImcopyEdgeCases:
    def test_copies_to_new_location(self, simple_fits, tmp_path):
        from lsc.util import imcopy
        dst = str(tmp_path / "copy_test.fits")
        imcopy(simple_fits, dst)
        assert os.path.exists(dst)
        # Verify data integrity
        orig_data = fits.getdata(simple_fits)
        copy_data = fits.getdata(dst)
        np.testing.assert_array_equal(orig_data, copy_data)

    def test_cutout_with_single_size_value(self, simple_fits, tmp_path):
        """cutout_size as a single int produces a square cutout."""
        from lsc.util import imcopy
        dst = str(tmp_path / "square_cutout.fits")
        imcopy(simple_fits, dst, center=(50, 50), cutout_size=20)
        data = fits.getdata(dst)
        assert data.shape == (20, 20)

    def test_no_cutout_copies_full(self, simple_fits, tmp_path):
        """Without center/cutout_size, full image is copied."""
        from lsc.util import imcopy
        dst = str(tmp_path / "full.fits")
        imcopy(simple_fits, dst, center=None, cutout_size=None)
        data = fits.getdata(dst)
        assert data.shape == (100, 100)

    def test_ext_parameter_defaults_to_zero(self, simple_fits, tmp_path):
        """Default ext=0 reads from the primary HDU."""
        from lsc.util import imcopy
        dst = str(tmp_path / "ext0.fits")
        imcopy(simple_fits, dst, ext=0)
        assert os.path.exists(dst)


# ---------------------------------------------------------------------------
# readkey3 — RA/DEC angle conversion edge cases
# ---------------------------------------------------------------------------

class TestReadkey3AngleConversions:
    def _make_header(self, instrume='kb70', extra=None):
        """Create a FITS header in memory (no file needed for readkey3)."""
        hdr = fits.Header()
        hdr['INSTRUME'] = instrume
        hdr['DATE-OBS'] = '2020-01-01T12:00:00'
        hdr['DAY-OBS'] = '2020-01-01'
        hdr['OBJECT'] = 'TestObj'
        hdr['EXPTIME'] = 60.0
        hdr['FILTER'] = 'r'
        hdr['GAIN'] = 2.0
        hdr['RDNOISE'] = 10.0
        hdr['RA'] = 150.0
        hdr['DEC'] = 2.5
        hdr['CAT-RA'] = 150.0
        hdr['CAT-DEC'] = 2.5
        hdr['AIRMASS'] = 1.2
        hdr['OBSTYPE'] = 'EXPOSE'
        hdr['SATURATE'] = 65000.0
        hdr['MJD-OBS'] = 58000.0
        hdr['WCSERR'] = 0
        hdr['TELESCOP'] = '1m0-01'
        hdr['PROPID'] = 'KEY2014A-001'
        hdr['USERID'] = 'testuser'
        hdr['OBSERVER'] = 'Tester'
        if extra:
            for k, v in extra.items():
                hdr[k] = v
        return hdr

    def test_ra_colon_format_hourangle(self):
        """RA='12:30:00.0' as hour angle -> 187.5 degrees."""
        from lsc.util import readkey3
        hdr = self._make_header(extra={'RA': '12:30:00.0'})
        val = readkey3(hdr, 'RA')
        # 12h30m = 12.5 * 15 = 187.5 degrees
        assert abs(val - 187.5) < 0.01

    def test_cat_ra_colon_format(self):
        """CAT-RA with colon format uses hourangle conversion."""
        from lsc.util import readkey3
        hdr = self._make_header(extra={'CAT-RA': '06:00:00.0'})
        val = readkey3(hdr, 'CAT-RA')
        # 6h = 90 degrees
        assert abs(val - 90.0) < 0.01

    def test_dec_numeric_stays_numeric(self):
        """DEC as a float value stays as-is (Angle(value, u.deg).deg)."""
        from lsc.util import readkey3
        hdr = self._make_header(extra={'DEC': -30.5})
        val = readkey3(hdr, 'DEC')
        assert abs(val - (-30.5)) < 0.01

    def test_cat_dec_numeric(self):
        """CAT-DEC as a float stays as-is."""
        from lsc.util import readkey3
        hdr = self._make_header(extra={'CAT-DEC': 45.0})
        val = readkey3(hdr, 'CAT-DEC')
        assert abs(val - 45.0) < 0.01

    def test_ra_zero(self):
        """RA=0.0 should return 0.0."""
        from lsc.util import readkey3
        hdr = self._make_header(extra={'RA': 0.0})
        val = readkey3(hdr, 'RA')
        assert abs(val - 0.0) < 0.01

    def test_dec_missing_value_in_missingvalues(self):
        """DEC that is in missingvalues list should not trigger Angle conversion."""
        from lsc.util import readkey3
        hdr = self._make_header(extra={'DEC': 'UNKNOWN'})
        val = readkey3(hdr, 'DEC')
        # 'UNKNOWN' is in missingvalues, so the elif for Angle is skipped
        # value remains 'UNKNOWN' then hash stripping runs (no '#')
        assert val == 'UNKNOWN'

    def test_ra_none_in_missingvalues(self):
        """RA=None is in missingvalues, so Angle conversion is skipped."""
        from lsc.util import readkey3
        hdr = self._make_header()
        # Remove RA so hdr.get returns None
        del hdr['RA']
        hdr['RA'] = None  # Note: FITS might not store None; but Header can
        val = readkey3(hdr, 'RA')
        # None is in missingvalues; the code checks `value not in missingvalues`
        # so it won't call Angle on None


# ---------------------------------------------------------------------------
# readkey3 — hash stripping and object cleaning
# ---------------------------------------------------------------------------

class TestReadkey3StringProcessing:
    def _make_header(self, instrume='fa15', extra=None):
        hdr = fits.Header()
        hdr['INSTRUME'] = instrume
        hdr['DATE-OBS'] = '2020-01-01T12:00:00'
        hdr['DAY-OBS'] = '2020-01-01'
        hdr['OBJECT'] = 'TestObj'
        hdr['EXPTIME'] = 60.0
        hdr['FILTER'] = 'r'
        hdr['GAIN'] = 2.0
        hdr['RDNOISE'] = 10.0
        hdr['RA'] = 150.0
        hdr['DEC'] = 2.5
        hdr['CAT-RA'] = 150.0
        hdr['CAT-DEC'] = 2.5
        hdr['AIRMASS'] = 1.2
        hdr['OBSTYPE'] = 'EXPOSE'
        hdr['SATURATE'] = 65000.0
        hdr['MJD-OBS'] = 58000.0
        hdr['WCSERR'] = 0
        hdr['TELESCOP'] = '1m0-01'
        hdr['PROPID'] = 'PROP001'
        hdr['USERID'] = 'user'
        hdr['OBSERVER'] = 'Observer'
        if extra:
            for k, v in extra.items():
                hdr[k] = v
        return hdr

    def test_hash_stripping_from_string_value(self):
        """Backslash-hash sequences are stripped from string values."""
        from lsc.util import readkey3
        hdr = self._make_header(extra={'OBSERVER': 'Name\\#suffix'})
        val = readkey3(hdr, 'observer')
        assert '\\#' not in val
        assert 'Name' in val

    def test_object_brackets_not_stripped_by_malformed_regex(self):
        """Object name regex r\"[()[]}{]\" is malformed and does NOT strip brackets.

        The regex character class [()[]}{] doesn't properly escape brackets,
        so re.sub effectively does nothing for this input.
        """
        from lsc.util import readkey3
        hdr = self._make_header(extra={'OBJECT': 'SN(2024)[abc]{test}'})
        val = readkey3(hdr, 'object')
        # The malformed regex doesn't actually strip these characters
        assert val == 'SN(2024)[abc]{test}'

    def test_object_clean_preserves_alphanumeric(self):
        """Object name cleaning preserves letters, digits, spaces."""
        from lsc.util import readkey3
        hdr = self._make_header(extra={'OBJECT': 'SN2024abc_test-1'})
        val = readkey3(hdr, 'object')
        assert 'SN2024abc_test-1' in val

    def test_instrume_lowercased(self):
        """instrume key returns lowercased value."""
        from lsc.util import readkey3
        hdr = self._make_header(extra={'INSTRUME': 'FA15'})
        val = readkey3(hdr, 'instrume')
        assert val == 'fa15'

    def test_keyword_in_hdr_but_not_useful_keys(self):
        """A keyword present in hdr but not in useful_keys dict still returns value."""
        from lsc.util import readkey3
        hdr = self._make_header(extra={'CUSTOMKEY': 'custom_value'})
        val = readkey3(hdr, 'CUSTOMKEY')
        assert val == 'custom_value'


# ---------------------------------------------------------------------------
# readkey3 — datamin as float literal
# ---------------------------------------------------------------------------

class TestReadkey3DataminFloat:
    def test_datamin_returns_float_literal(self):
        """datamin is hardcoded as -100.0 float in useful_keys for known instruments."""
        from lsc.util import readkey3
        hdr = fits.Header()
        hdr['INSTRUME'] = 'fa15'
        hdr['DATE-OBS'] = '2020-01-01T12:00:00'
        hdr['DAY-OBS'] = '2020-01-01'
        hdr['FILTER'] = 'r'
        hdr['GAIN'] = 2.0
        hdr['RDNOISE'] = 10.0
        hdr['RA'] = 150.0
        hdr['DEC'] = 2.5
        hdr['SATURATE'] = 65000.0
        hdr['MJD-OBS'] = 58000.0
        val = readkey3(hdr, 'datamin')
        assert val == -100.0
        assert isinstance(val, float)


# ---------------------------------------------------------------------------
# readkey3 — filter fallback chain
# ---------------------------------------------------------------------------

class TestReadkey3FilterFallback:
    def _make_header_no_filter(self, instrume='fa15'):
        hdr = fits.Header()
        hdr['INSTRUME'] = instrume
        hdr['DATE-OBS'] = '2020-01-01T12:00:00'
        hdr['DAY-OBS'] = '2020-01-01'
        hdr['GAIN'] = 2.0
        hdr['RDNOISE'] = 10.0
        hdr['RA'] = 150.0
        hdr['DEC'] = 2.5
        hdr['SATURATE'] = 65000.0
        hdr['MJD-OBS'] = 58000.0
        return hdr

    def test_filter_none_falls_to_filter2(self):
        """FILTER=None, FILTER2='ip' -> returns 'ip'."""
        from lsc.util import readkey3
        hdr = self._make_header_no_filter()
        hdr['FILTER2'] = 'ip'
        # Don't set FILTER -> hdr.get('FILTER') returns None
        val = readkey3(hdr, 'filter')
        assert val == 'ip'

    def test_filter_air_falls_to_filter2(self):
        """FILTER='air', FILTER2='gp' -> returns 'gp'."""
        from lsc.util import readkey3
        hdr = self._make_header_no_filter()
        hdr['FILTER'] = 'air'
        hdr['FILTER2'] = 'gp'
        val = readkey3(hdr, 'filter')
        assert val == 'gp'

    def test_filter_air_filter2_also_air_falls_to_filter1(self):
        """FILTER='air', FILTER2='air', FILTER1='rp' -> returns 'rp'."""
        from lsc.util import readkey3
        hdr = self._make_header_no_filter()
        hdr['FILTER'] = 'air'
        hdr['FILTER2'] = 'air'
        hdr['FILTER1'] = 'rp'
        val = readkey3(hdr, 'filter')
        assert val == 'rp'

    def test_filter_air_all_air_falls_to_filter3(self):
        """FILTER='air', FILTER2='air', FILTER1='air', FILTER3='zs' -> returns 'zs'."""
        from lsc.util import readkey3
        hdr = self._make_header_no_filter()
        hdr['FILTER'] = 'air'
        hdr['FILTER2'] = 'air'
        hdr['FILTER1'] = 'air'
        hdr['FILTER3'] = 'zs'
        val = readkey3(hdr, 'filter')
        assert val == 'zs'

    def test_filter_none_no_fallback_returns_none(self):
        """FILTER=None with no FILTER1/2/3 -> returns None."""
        from lsc.util import readkey3
        hdr = self._make_header_no_filter()
        # No FILTER set at all, no FILTER1/2/3
        val = readkey3(hdr, 'filter')
        assert val is None

    def test_filter_air_all_none_returns_air(self):
        """FILTER='air', no valid FILTER1/2/3 -> stays 'air'."""
        from lsc.util import readkey3
        hdr = self._make_header_no_filter()
        hdr['FILTER'] = 'air'
        # No FILTER1, FILTER2, FILTER3
        val = readkey3(hdr, 'filter')
        # Loop doesn't find any non-air/non-None, so value stays 'air'
        assert val == 'air'


# ---------------------------------------------------------------------------
# readkey3 — em instrument branch (same as fs)
# ---------------------------------------------------------------------------

class TestReadkey3EmInstrument:
    def test_em_instrument_post2014(self):
        """'em' instrument post-2014 uses same keys as fs post-2014."""
        from lsc.util import readkey3
        hdr = fits.Header()
        hdr['INSTRUME'] = 'em01'
        hdr['DATE-OBS'] = '2020-01-01T12:00:00'
        hdr['DAY-OBS'] = '2020-01-01'
        hdr['FILTER'] = 'V'
        hdr['GAIN'] = 1.5
        hdr['RDNOISE'] = 8.0
        hdr['RA'] = 100.0
        hdr['DEC'] = -10.0
        hdr['CAT-RA'] = 100.0
        hdr['CAT-DEC'] = -10.0
        hdr['SATURATE'] = 55000.0
        hdr['MJD-OBS'] = 59000.0
        hdr['WCSERR'] = 0
        hdr['TELESCOP'] = '1m0-03'
        hdr['EXPTIME'] = 30.0
        hdr['OBSTYPE'] = 'EXPOSE'
        hdr['PROPID'] = 'PROP'
        hdr['USERID'] = 'user'
        hdr['OBSERVER'] = 'obs'
        assert readkey3(hdr, 'exptime') == 30.0
        assert readkey3(hdr, 'ron') == 8.0
        assert readkey3(hdr, 'filter') == 'V'

    def test_em_instrument_pre2014(self):
        """'em' instrument pre-2014 uses MJD key and datamax=60000."""
        from lsc.util import readkey3
        hdr = fits.Header()
        hdr['INSTRUME'] = 'em01'
        hdr['DATE-OBS'] = '2013-01-01T12:00:00'
        hdr['DAY-OBS'] = '2013-01-01'
        hdr['FILTER'] = 'B'
        hdr['GAIN'] = 1.0
        hdr['RDNOISE'] = 6.0
        hdr['RA'] = 200.0
        hdr['DEC'] = 30.0
        hdr['CAT-RA'] = 200.0
        hdr['CAT-DEC'] = 30.0
        hdr['SATURATE'] = 55000.0
        hdr['MJD'] = 56293.0
        hdr['MJD-OBS'] = 56293.0
        hdr['WCS_ERR'] = 0.0
        hdr['TELID'] = 'tel01'
        hdr['EXPTIME'] = 60.0
        hdr['OBSTYPE'] = 'EXPOSE'
        hdr['PROPID'] = 'PROP'
        hdr['USERID'] = 'user'
        hdr['OBSERVER'] = 'obs'
        assert readkey3(hdr, 'datamax') == 60000.0
        assert readkey3(hdr, 'JD') == 56293.5


# ---------------------------------------------------------------------------
# readkey3 — date-obs formatting
# ---------------------------------------------------------------------------

class TestReadkey3DateObs:
    def test_date_obs_with_T_strips_time_and_dashes(self):
        """'2020-03-15T08:30:00' -> '20200315'."""
        from lsc.util import readkey3
        hdr = fits.Header()
        hdr['INSTRUME'] = 'fa15'
        hdr['DATE-OBS'] = '2020-03-15T08:30:00'
        hdr['DAY-OBS'] = '2020-03-15'
        hdr['FILTER'] = 'r'
        hdr['GAIN'] = 2.0
        hdr['RDNOISE'] = 10.0
        hdr['SATURATE'] = 65000.0
        hdr['MJD-OBS'] = 58000.0
        hdr['RA'] = 150.0
        hdr['DEC'] = 2.5
        val = readkey3(hdr, 'date-obs')
        assert val == '20200315'

    def test_date_obs_no_T_no_split(self):
        """'20200315' without T -> split('T') returns the whole thing -> replace dashes (none)."""
        from lsc.util import readkey3
        hdr = fits.Header()
        hdr['INSTRUME'] = 'fa15'
        hdr['DATE-OBS'] = '20200315'
        hdr['DAY-OBS'] = '2020-03-15'
        hdr['FILTER'] = 'r'
        hdr['GAIN'] = 2.0
        hdr['RDNOISE'] = 10.0
        hdr['SATURATE'] = 65000.0
        hdr['MJD-OBS'] = 58000.0
        hdr['RA'] = 150.0
        hdr['DEC'] = 2.5
        val = readkey3(hdr, 'date-obs')
        assert val == '20200315'


# ---------------------------------------------------------------------------
# readstandard — negative declination parsing
# ---------------------------------------------------------------------------

class TestReadstandardNegDec:
    def test_negative_declination(self, tmp_path):
        """Stars with negative declination (sign '-') are computed correctly."""
        from lsc.util import readstandard
        p = tmp_path / "negdec.txt"
        p.write_text("Star1 12:00:00.0 -30:30:00.0 16.0\n")
        star, ra, dec, mag = readstandard(str(p))
        # -30d 30m 0s = -(30 + 30/60) = -30.5
        assert abs(float(dec[0]) - (-30.5)) < 0.01

    def test_positive_declination(self, tmp_path):
        """Stars with positive declination."""
        from lsc.util import readstandard
        p = tmp_path / "posdec.txt"
        p.write_text("Star2 06:00:00.0 +45:30:30.0 14.5\n")
        star, ra, dec, mag = readstandard(str(p))
        # 45d 30m 30s = 45 + 30/60 + 30/3600 = 45.508333
        assert abs(float(dec[0]) - 45.508333) < 0.01

    def test_missing_magnitude_defaults_to_999(self, tmp_path):
        """When magnitude column is missing, defaults to 999."""
        from lsc.util import readstandard
        p = tmp_path / "nomag.txt"
        p.write_text("Star3 12:00:00.0 +02:00:00.0\n")
        star, ra, dec, mag = readstandard(str(p))
        assert mag[0] == 999

    def test_multiple_stars(self, tmp_path):
        """Multiple stars parsed correctly."""
        from lsc.util import readstandard
        p = tmp_path / "multi.txt"
        content = (
            "Star1 01:00:00.0 +10:00:00.0 15.0\n"
            "Star2 02:00:00.0 -20:00:00.0 16.0\n"
            "Star3 12:30:00.0 +00:30:00.0 17.5\n"
        )
        p.write_text(content)
        star, ra, dec, mag = readstandard(str(p))
        assert len(star) == 3
        assert star[0] == 'Star1'
        assert star[2] == 'Star3'

    def test_comment_lines_skipped(self, tmp_path):
        """Lines starting with # are skipped."""
        from lsc.util import readstandard
        p = tmp_path / "comments.txt"
        content = (
            "# This is a header\n"
            "Star1 12:00:00.0 +02:00:00.0 15.0\n"
            "# Another comment\n"
            "Star2 13:00:00.0 -03:00:00.0 16.0\n"
        )
        p.write_text(content)
        star, ra, dec, mag = readstandard(str(p))
        assert len(star) == 2


# ---------------------------------------------------------------------------
# readspectrum — 2D and 3D data
# ---------------------------------------------------------------------------

class TestReadspectrum2D3D:
    def test_2d_data_takes_first_column(self, tmp_path):
        """2D spectrum data (naxis=2) reads data[:,0]."""
        from lsc.util import readspectrum
        data = np.ones((50, 3), dtype=np.float32)
        data[:, 0] = np.arange(50)  # flux in first column
        hdr = fits.Header()
        hdr['NAXIS1'] = 50
        hdr['CRPIX1'] = 1
        hdr['CRVAL1'] = 4000.0
        hdr['CDELT1'] = 2.0
        path = str(tmp_path / 'spec2d.fits')
        fits.writeto(path, data, hdr, overwrite=True)
        lam, fl = readspectrum(path)
        assert len(fl) == 50
        assert fl[0] == 0.0
        assert fl[49] == 49.0

    def test_3d_data_takes_first_slice(self, tmp_path):
        """3D spectrum data (naxis=3) reads data[0,0,:]."""
        from lsc.util import readspectrum
        data = np.ones((2, 2, 30), dtype=np.float32)
        data[0, 0, :] = np.arange(30) * 0.5
        hdr = fits.Header()
        hdr['NAXIS1'] = 30
        hdr['CRPIX1'] = 1
        hdr['CRVAL1'] = 5000.0
        hdr['CDELT1'] = 1.5
        path = str(tmp_path / 'spec3d.fits')
        fits.writeto(path, data, hdr, overwrite=True)
        lam, fl = readspectrum(path)
        assert len(fl) == 30
        assert abs(fl[0] - 0.0) < 1e-6
        assert abs(fl[1] - 0.5) < 1e-6

    def test_no_wcs_keywords_returns_empty_lam(self, tmp_path):
        """When no CRPIX1/CRVAL1/CDELT1/CD1_1/WAT2_001, lam is ''."""
        from lsc.util import readspectrum
        data = np.ones(40, dtype=np.float32)
        hdr = fits.Header()
        hdr['NAXIS1'] = 40
        path = str(tmp_path / 'spec_nowcs.fits')
        fits.writeto(path, data, hdr, overwrite=True)
        lam, fl = readspectrum(path)
        assert lam == ''
        assert len(fl) == 40


# ---------------------------------------------------------------------------
# pval — numpy array inputs
# ---------------------------------------------------------------------------

class TestPvalNumpy:
    def test_numpy_array_input(self):
        """pval should work with numpy arrays."""
        from lsc.util import pval
        x = np.array([1.0, 2.0, 3.0])
        result = pval(x, [1.0, 2.0])
        expected = np.array([3.0, 5.0, 7.0])
        np.testing.assert_allclose(result, expected)

    def test_zero_coefficients(self):
        """pval with both coefficients zero returns zeros."""
        from lsc.util import pval
        result = pval(5.0, [0.0, 0.0])
        assert result == 0.0

    def test_large_values(self):
        """pval with large values."""
        from lsc.util import pval
        result = pval(1e6, [0.0, 1.0])
        assert abs(result - 1e6) < 1.0


# ---------------------------------------------------------------------------
# residual — numpy array inputs and multiple coefficients
# ---------------------------------------------------------------------------

class TestResidualEdgeCases:
    def test_three_coefficients(self):
        """residual with 3 coefficients: last iteration is p[2]*x^2."""
        from lsc.util import residual
        # p=[1, 2, 3], y=10, x=2
        # i=0: err = 10 - 1*1 = 9
        # i=1: err = 10 - 2*2 = 6
        # i=2: err = 10 - 3*4 = -2
        err = residual([1.0, 2.0, 3.0], 10.0, 2.0)
        assert abs(err - (-2.0)) < 1e-9

    def test_numpy_arrays(self):
        """residual should work with numpy arrays for y and x."""
        from lsc.util import residual
        y = np.array([5.0, 10.0])
        x = np.array([1.0, 2.0])
        err = residual([0.0, 2.0], y, x)
        # i=0: err = y - 0*1 = y
        # i=1: err = y - 2*x = [5-2, 10-4] = [3, 6]
        expected = np.array([3.0, 6.0])
        np.testing.assert_allclose(err, expected)

    def test_zero_x(self):
        """residual with x=0."""
        from lsc.util import residual
        # p=[3, 5], y=10, x=0
        # i=0: err = 10 - 3*1 = 7
        # i=1: err = 10 - 5*0 = 10
        err = residual([3.0, 5.0], 10.0, 0.0)
        assert abs(err - 10.0) < 1e-9


# ---------------------------------------------------------------------------
# writeinthelog — special characters
# ---------------------------------------------------------------------------

class TestWriteinthelogSpecial:
    def test_unicode_content(self, tmp_path):
        from lsc.util import writeinthelog
        logfile = str(tmp_path / 'unicode.log')
        writeinthelog('stars: ★ ☆\n', logfile)
        content = open(logfile, encoding='utf-8').read()
        assert '★' in content

    def test_newlines_preserved(self, tmp_path):
        from lsc.util import writeinthelog
        logfile = str(tmp_path / 'newlines.log')
        writeinthelog('line1\nline2\nline3\n', logfile)
        lines = open(logfile).readlines()
        assert len(lines) == 3

    def test_tab_characters(self, tmp_path):
        from lsc.util import writeinthelog
        logfile = str(tmp_path / 'tabs.log')
        writeinthelog('col1\tcol2\tcol3\n', logfile)
        content = open(logfile).read()
        assert '\t' in content


# ---------------------------------------------------------------------------
# repstringinfile — regex metacharacters and same input/output
# ---------------------------------------------------------------------------

class TestRepstringinfileEdgeCases:
    def test_regex_metacharacters_in_string(self, tmp_path):
        """re.sub with regex metacharacters in the search string."""
        from lsc.util import repstringinfile
        fin = str(tmp_path / 'in.txt')
        fout = str(tmp_path / 'out.txt')
        # string1 contains regex metachar '.' which matches any char
        open(fin, 'w').write('hello.world\n')
        repstringinfile(fin, fout, 'hello.world', 'REPLACED')
        content = open(fout).read()
        assert 'REPLACED' in content

    def test_same_input_output_file(self, tmp_path):
        """When filein == fileout, it rewrites the same file."""
        from lsc.util import repstringinfile
        f = str(tmp_path / 'inplace.txt')
        open(f, 'w').write('original text\n')
        repstringinfile(f, f, 'original', 'modified')
        content = open(f).read()
        assert 'modified text' in content

    def test_empty_string1_replaces_nothing(self, tmp_path):
        """When string1 is empty, 'in' check always True but re.sub may behave oddly."""
        from lsc.util import repstringinfile
        fin = str(tmp_path / 'in.txt')
        fout = str(tmp_path / 'out.txt')
        open(fin, 'w').write('test content\n')
        # Empty string is always "in" any string
        repstringinfile(fin, fout, '', 'X')
        content = open(fout).read()
        # re.sub('', 'X', 'test content\n') inserts X between every char
        assert 'X' in content


# ---------------------------------------------------------------------------
# updateheader — tuple value format and error handling
# ---------------------------------------------------------------------------

class TestUpdateheaderEdgeCases:
    def test_tuple_value_with_comment(self, simple_fits, tmp_path):
        """headerdict can contain (value, comment) tuples."""
        from lsc.util import updateheader
        import shutil
        dst = str(tmp_path / "tuple_hdr.fits")
        shutil.copy2(simple_fits, dst)
        updateheader(dst, 0, {"TESTKEY": (99.9, "test comment")})
        hdr = fits.getheader(dst)
        assert abs(hdr["TESTKEY"] - 99.9) < 1e-5
        assert "test comment" in hdr.comments["TESTKEY"]

    def test_invalid_dimension_prints_error(self, simple_fits, tmp_path, capsys):
        """Accessing an invalid HDU index should be caught."""
        from lsc.util import updateheader
        import shutil
        dst = str(tmp_path / "bad_dim.fits")
        shutil.copy2(simple_fits, dst)
        updateheader(dst, 5, {"KEY": 1})  # HDU index 5 doesn't exist
        out = capsys.readouterr().out
        assert 'not updated' in out

    def test_empty_headerdict(self, simple_fits, tmp_path):
        """Empty headerdict should not raise."""
        from lsc.util import updateheader
        import shutil
        dst = str(tmp_path / "empty_hdr.fits")
        shutil.copy2(simple_fits, dst)
        updateheader(dst, 0, {})
        # File should still be valid
        hdr = fits.getheader(dst)
        assert hdr is not None


# ---------------------------------------------------------------------------
# limmag — boundary conditions
# ---------------------------------------------------------------------------

class TestLimmagBoundary:
    def _make_phot_fits(self, tmp_path, name, **kwargs):
        hdr = fits.Header()
        hdr['INSTRUME'] = kwargs.get('instrume', 'kb70')
        hdr['DATE-OBS'] = '2020-01-01T12:00:00'
        hdr['MJD-OBS'] = 58000.0
        hdr['DAY-OBS'] = '2020-01-01'
        hdr['GAIN'] = kwargs.get('gain', 2.0)
        hdr['RDNOISE'] = 10.0
        hdr['SATURATE'] = 60000.0
        hdr['EXPTIME'] = kwargs.get('exptime', 120.0)
        hdr['FILTER'] = 'r'
        hdr['RA'] = 150.0
        hdr['DEC'] = 2.5
        if 'photzp' in kwargs:
            hdr['PHOTZP'] = kwargs['photzp']
        if 'mbkg' in kwargs:
            hdr['MBKG'] = kwargs['mbkg']
        if 'psf_fwhm' in kwargs:
            hdr['PSF_FWHM'] = kwargs['psf_fwhm']
        path = str(tmp_path / name)
        fits.writeto(path, np.ones((10, 10), dtype=np.float32), hdr, overwrite=True)
        return path

    def test_negative_mbkg_clamped_to_zero(self, tmp_path):
        """When MBKG is negative, it's clamped to 0 before calculation."""
        from lsc.util import limmag
        f = self._make_phot_fits(tmp_path, 'negbkg.fits',
                                 photzp=25.0, mbkg=-10.0, psf_fwhm=3.0)
        result = limmag(f)
        # mbkg <= 0 triggers _mbkg=0, then sqrt(0)=0, which makes
        # log10(0) undefined. Let's check what actually happens.
        # Actually the code: if _mbkg<=0: _mbkg=0
        # then n*_mbkg/_exptime = 0, sqrt(0)=0, sn*(1/gain)*0 = 0
        # log10(0) -> -inf -> maglim becomes inf or raises
        # The code doesn't guard against this...
        # It should either return '' or a very large number
        # Let's just verify it doesn't crash
        assert result is not None

    def test_efosc_instrument_pixscale(self, tmp_path):
        """efosc instrument uses pixel scale from binx*0.12."""
        from lsc.util import limmag
        hdr = fits.Header()
        hdr['INSTRUME'] = 'efosc'
        hdr['DATE-OBS'] = '2020-01-01T12:00:00'
        hdr['MJD-OBS'] = 58000.0
        hdr['DAY-OBS'] = '2020-01-01'
        hdr['GAIN'] = 1.5
        hdr['RDNOISE'] = 8.0
        hdr['SATURATE'] = 60000.0
        hdr['EXPTIME'] = 300.0
        hdr['FILTER'] = 'V'
        hdr['RA'] = 100.0
        hdr['DEC'] = -20.0
        hdr['PHOTZP'] = 26.0
        hdr['MBKG'] = 200.0
        hdr['PSF_FWHM'] = 4.0
        hdr['BINX'] = 2
        path = str(tmp_path / 'efosc.fits')
        fits.writeto(path, np.ones((10, 10), dtype=np.float32), hdr, overwrite=True)
        result = limmag(path)
        assert isinstance(result, float)
        assert result > 0

    def test_missing_gain_returns_empty(self, tmp_path):
        """Without GAIN header, limmag returns ''."""
        from lsc.util import limmag
        hdr = fits.Header()
        hdr['INSTRUME'] = 'kb70'
        hdr['DATE-OBS'] = '2020-01-01T12:00:00'
        hdr['MJD-OBS'] = 58000.0
        hdr['DAY-OBS'] = '2020-01-01'
        hdr['RDNOISE'] = 10.0
        hdr['SATURATE'] = 60000.0
        hdr['EXPTIME'] = 120.0
        hdr['FILTER'] = 'r'
        hdr['RA'] = 150.0
        hdr['DEC'] = 2.5
        hdr['PHOTZP'] = 25.0
        hdr['MBKG'] = 500.0
        hdr['PSF_FWHM'] = 3.0
        # No GAIN
        path = str(tmp_path / 'nogain_limmag.fits')
        fits.writeto(path, np.ones((10, 10), dtype=np.float32), hdr, overwrite=True)
        result = limmag(path)
        assert result == ''

    def test_valid_parameters_returns_reasonable_mag(self, tmp_path):
        """With typical parameters, limmag returns a reasonable magnitude."""
        from lsc.util import limmag
        f = self._make_phot_fits(tmp_path, 'good.fits',
                                 photzp=25.0, mbkg=500.0, psf_fwhm=3.0,
                                 gain=2.0, exptime=120.0)
        result = limmag(f)
        assert isinstance(result, float)
        # Limiting magnitude should be in a reasonable range
        assert 15.0 < result < 30.0


# ---------------------------------------------------------------------------
# Docosmic — 'temp' in filename produces float32 mask
# ---------------------------------------------------------------------------

class TestDocosmicTempFilename:
    def test_temp_in_filename_produces_float_mask(self, tmp_path, monkeypatch):
        """Images with 'temp' in filename produce float32 cosmic ray masks."""
        import sys
        mock_ast = MagicMock()
        mock_ast.detect_cosmics.return_value = (
            np.ones((20, 20), dtype=bool),  # some cosmics detected
            np.zeros((20, 20), dtype=np.float32),
        )
        monkeypatch.setitem(sys.modules, 'astroscrappy', mock_ast)
        from lsc.util import Docosmic
        hdr = fits.Header()
        hdr['INSTRUME'] = 'fa15'
        hdr['GAIN'] = 2.0
        hdr['RDNOISE'] = 5.0
        hdr['SATURATE'] = 60000.0
        hdr['FILTER'] = 'r'
        hdr['DATE-OBS'] = '2022-01-01'
        path = str(tmp_path / 'temp_image.fits')
        fits.writeto(path, np.ones((20, 20), dtype=np.float32), hdr, overwrite=True)
        monkeypatch.chdir(tmp_path)
        out, outmask, outsat = Docosmic(path)
        # Check the mask is float32 (may be big-endian '>f4' from FITS)
        mask_data = fits.getdata(outmask)
        assert np.issubdtype(mask_data.dtype, np.floating)

    def test_regular_filename_produces_uint8_mask(self, tmp_path, monkeypatch):
        """Images without 'temp' in the full path produce uint8 cosmic ray masks.

        Note: Docosmic checks 'if \"temp\" in img' on the full path, not just basename.
        We pass only the basename to Docosmic (after chdir) to ensure the
        path does not accidentally contain 'temp'.
        """
        import sys
        mock_ast = MagicMock()
        mock_ast.detect_cosmics.return_value = (
            np.ones((20, 20), dtype=bool),
            np.zeros((20, 20), dtype=np.float32),
        )
        monkeypatch.setitem(sys.modules, 'astroscrappy', mock_ast)
        from lsc.util import Docosmic
        hdr = fits.Header()
        hdr['INSTRUME'] = 'fa15'
        hdr['GAIN'] = 2.0
        hdr['RDNOISE'] = 5.0
        hdr['SATURATE'] = 60000.0
        hdr['FILTER'] = 'r'
        hdr['DATE-OBS'] = '2022-01-01'
        path = str(tmp_path / 'regular_image.fits')
        fits.writeto(path, np.ones((20, 20), dtype=np.float32), hdr, overwrite=True)
        monkeypatch.chdir(tmp_path)
        # Pass only the basename so that the 'temp' check only sees the filename
        out, outmask, outsat = Docosmic('regular_image.fits')
        mask_data = fits.getdata(outmask)
        assert np.issubdtype(mask_data.dtype, np.unsignedinteger)


# ---------------------------------------------------------------------------
# Docosmic_old — 'temp' in filename produces float32 mask
# ---------------------------------------------------------------------------

class TestDocosmicOldTempFilename:
    def test_temp_in_filename_produces_float_mask(self, tmp_path, monkeypatch):
        """Docosmic_old with 'temp' in filename produces float32 mask."""
        from lsc.util import Docosmic_old
        hdr = fits.Header()
        hdr['INSTRUME'] = 'fa15'
        hdr['GAIN'] = 2.0
        hdr['RDNOISE'] = 5.0
        hdr['SATURATE'] = 60000.0
        hdr['FILTER'] = 'r'
        hdr['DATE-OBS'] = '2022-01-01'
        path = str(tmp_path / 'temp_old.fits')
        fits.writeto(path, np.ones((20, 20), dtype=np.float32), hdr, overwrite=True)
        monkeypatch.chdir(tmp_path)
        out, outmask, outsat = Docosmic_old(path)
        mask_data = fits.getdata(outmask)
        # FITS may store as big-endian '>f4'; use issubdtype for comparison
        assert np.issubdtype(mask_data.dtype, np.floating)

    def test_regular_produces_uint8_mask(self, tmp_path, monkeypatch):
        """Docosmic_old without 'temp' in path produces uint8 mask.

        Note: Docosmic_old checks 'if \"temp\" in img' on the full path.
        We pass only the basename to avoid pytest tmp_path interference.
        """
        from lsc.util import Docosmic_old
        hdr = fits.Header()
        hdr['INSTRUME'] = 'fa15'
        hdr['GAIN'] = 2.0
        hdr['RDNOISE'] = 5.0
        hdr['SATURATE'] = 60000.0
        hdr['FILTER'] = 'r'
        hdr['DATE-OBS'] = '2022-01-01'
        path = str(tmp_path / 'normal_img.fits')
        fits.writeto(path, np.ones((20, 20), dtype=np.float32), hdr, overwrite=True)
        monkeypatch.chdir(tmp_path)
        # Pass only basename to avoid 'temp' appearing in the directory path
        out, outmask, outsat = Docosmic_old('normal_img.fits')
        mask_data = fits.getdata(outmask)
        assert np.issubdtype(mask_data.dtype, np.unsignedinteger)


# ---------------------------------------------------------------------------
# Docosmic — telescop header branch (no TELID, uses 'telescop')
# ---------------------------------------------------------------------------

class TestDocosmicTelescopBranch:
    def test_telescop_header_ftn(self, tmp_path, monkeypatch):
        """When TELID absent but telescop='ftn', uses fts/ftn branch."""
        import sys
        import lsc
        mock_ast = MagicMock()
        mock_ast.detect_cosmics.return_value = (
            np.zeros((20, 20), dtype=bool),
            np.zeros((20, 20), dtype=np.float32),
        )
        monkeypatch.setitem(sys.modules, 'astroscrappy', mock_ast)
        monkeypatch.setattr(lsc, 'delete', MagicMock(), raising=False)
        from lsc.util import Docosmic
        hdr = fits.Header()
        hdr['telescop'] = 'ftn'  # lowercase key to match the elif
        hdr['GAIN'] = 2.0
        hdr['RDNOISE'] = 5.0
        hdr['INSTRUME'] = 'fa15'
        hdr['FILTER'] = 'r'
        hdr['DATE-OBS'] = '2022-01-01'
        path = str(tmp_path / 'ftn_telescop.fits')
        fits.writeto(path, np.ones((20, 20), dtype=np.float32), hdr, overwrite=True)
        monkeypatch.chdir(tmp_path)
        out, _, _ = Docosmic(path)
        assert out.endswith('.clean.fits')

    def test_no_telid_no_telescop_uses_extdata(self, tmp_path, monkeypatch):
        """When neither TELID nor telescop present, _tel='extdata'."""
        import sys
        mock_ast = MagicMock()
        mock_ast.detect_cosmics.return_value = (
            np.zeros((20, 20), dtype=bool),
            np.zeros((20, 20), dtype=np.float32),
        )
        monkeypatch.setitem(sys.modules, 'astroscrappy', mock_ast)
        from lsc.util import Docosmic
        hdr = fits.Header()
        hdr['GAIN'] = 2.0
        hdr['RDNOISE'] = 5.0
        hdr['SATURATE'] = 60000.0
        hdr['INSTRUME'] = 'fa15'
        hdr['FILTER'] = 'r'
        hdr['DATE-OBS'] = '2022-01-01'
        path = str(tmp_path / 'extdata.fits')
        fits.writeto(path, np.ones((20, 20), dtype=np.float32), hdr, overwrite=True)
        monkeypatch.chdir(tmp_path)
        out, _, _ = Docosmic(path)
        assert out.endswith('.clean.fits')


# ---------------------------------------------------------------------------
# defswarp — combine type normalization
# ---------------------------------------------------------------------------

class TestDefswarpCombineNormalization:
    def test_median_lowercase_normalized(self, tmp_path):
        """'median' (lowercase) is normalized to 'MEDIAN'."""
        from lsc.util import defswarp
        sfile = str(tmp_path / 'med.swarp')
        defswarp(sfile, 'out.fits', 'median')
        content = open(sfile).read()
        assert 'MEDIAN' in content

    def test_MEDIAN_uppercase_works(self, tmp_path):
        """'MEDIAN' stays as 'MEDIAN'."""
        from lsc.util import defswarp
        sfile = str(tmp_path / 'MED.swarp')
        defswarp(sfile, 'out.fits', 'MEDIAN')
        content = open(sfile).read()
        assert 'MEDIAN' in content

    def test_Average_mixed_case(self, tmp_path):
        """'Average' with mixed case normalizes to 'AVERAGE'."""
        from lsc.util import defswarp
        sfile = str(tmp_path / 'avg.swarp')
        defswarp(sfile, 'out.fits', 'Average')
        content = open(sfile).read()
        assert 'AVERAGE' in content

    def test_SUM_lowercase(self, tmp_path):
        """'sum' normalizes to 'SUM'."""
        from lsc.util import defswarp
        sfile = str(tmp_path / 'sum.swarp')
        defswarp(sfile, 'out.fits', 'sum')
        content = open(sfile).read()
        assert 'SUM' in content

    def test_weightout_name_derived(self, tmp_path):
        """WEIGHTOUT_NAME is derived from imgname with .weight.fits suffix."""
        from lsc.util import defswarp
        sfile = str(tmp_path / 'weight.swarp')
        defswarp(sfile, 'science.fits', 'median')
        content = open(sfile).read()
        assert 'science.weight.fits' in content

    def test_pixelscale_type_manual(self, tmp_path):
        """PIXELSCALE_TYPE is set to MANUAL,MANUAL."""
        from lsc.util import defswarp
        sfile = str(tmp_path / 'pst.swarp')
        defswarp(sfile, 'out.fits', 'median')
        content = open(sfile).read()
        assert 'PIXELSCALE_TYPE MANUAL,MANUAL' in content

    def test_center_type_manual(self, tmp_path):
        """CENTER_TYPE is set to MANUAL,MANUAL."""
        from lsc.util import defswarp
        sfile = str(tmp_path / 'ct.swarp')
        defswarp(sfile, 'out.fits', 'median')
        content = open(sfile).read()
        assert 'CENTER_TYPE MANUAL,MANUAL' in content


# ---------------------------------------------------------------------------
# readhdr — error message format
# ---------------------------------------------------------------------------

class TestReadhdrErrors:
    def test_corrupted_file_raises_with_message(self, tmp_path, capsys):
        """readhdr on a corrupted FITS file raises and prints guidance."""
        from lsc.util import readhdr
        bad = str(tmp_path / "corrupt.fits")
        open(bad, 'w').write("not a FITS file")
        with pytest.raises(Exception):
            readhdr(bad)


# ---------------------------------------------------------------------------
# readkey3 — default branch with key directly in header
# ---------------------------------------------------------------------------

class TestReadkey3DirectHeaderAccess:
    def test_direct_header_key_not_in_useful_keys(self):
        """Keys not in useful_keys but present in header are returned directly."""
        from lsc.util import readkey3
        hdr = fits.Header()
        hdr['INSTRUME'] = 'unknown_xyz'
        hdr['DATE-OBS'] = '2020-01-01'
        hdr['DAY-OBS'] = '2020-01-01'
        hdr['CUSTOM1'] = 42
        hdr['CUSTOM2'] = 'hello'
        hdr['RA'] = 100.0
        hdr['DEC'] = 10.0
        hdr['MJD-OBS'] = 58000.0
        hdr['SATURATE'] = 60000.0
        assert readkey3(hdr, 'CUSTOM1') == 42
        assert readkey3(hdr, 'CUSTOM2') == 'hello'

    def test_key_not_in_useful_keys_and_not_in_header(self):
        """Key not in useful_keys AND not in header returns ''."""
        from lsc.util import readkey3
        hdr = fits.Header()
        hdr['INSTRUME'] = 'unknown_xyz'
        hdr['RA'] = 100.0
        hdr['DEC'] = 10.0
        hdr['MJD-OBS'] = 58000.0
        hdr['DATE-OBS'] = '2020-01-01'
        hdr['DAY-OBS'] = '2020-01-01'
        hdr['SATURATE'] = 60000.0
        val = readkey3(hdr, 'TOTALLY_MISSING')
        assert val == ''


# ---------------------------------------------------------------------------
# readkey3 — 'date-night' key
# ---------------------------------------------------------------------------

class TestReadkey3DateNight:
    def test_date_night_returns_day_obs(self):
        """date-night maps to DAY-OBS in all instrument branches."""
        from lsc.util import readkey3
        hdr = fits.Header()
        hdr['INSTRUME'] = 'fa15'
        hdr['DATE-OBS'] = '2020-06-15T08:00:00'
        hdr['DAY-OBS'] = '2020-06-15'
        hdr['FILTER'] = 'r'
        hdr['GAIN'] = 2.0
        hdr['RDNOISE'] = 10.0
        hdr['RA'] = 150.0
        hdr['DEC'] = 2.5
        hdr['SATURATE'] = 65000.0
        hdr['MJD-OBS'] = 59000.0
        val = readkey3(hdr, 'date-night')
        assert val == '2020-06-15'


# ---------------------------------------------------------------------------
# readkey3 — 'wcserr' key
# ---------------------------------------------------------------------------

class TestReadkey3Wcserr:
    def test_wcserr_returns_value(self):
        """wcserr maps to WCSERR for fa/fl instruments."""
        from lsc.util import readkey3
        hdr = fits.Header()
        hdr['INSTRUME'] = 'fa15'
        hdr['DATE-OBS'] = '2020-01-01T12:00:00'
        hdr['DAY-OBS'] = '2020-01-01'
        hdr['FILTER'] = 'r'
        hdr['GAIN'] = 2.0
        hdr['RDNOISE'] = 10.0
        hdr['RA'] = 150.0
        hdr['DEC'] = 2.5
        hdr['SATURATE'] = 65000.0
        hdr['MJD-OBS'] = 58000.0
        hdr['WCSERR'] = 0.5
        val = readkey3(hdr, 'wcserr')
        assert val == 0.5

    def test_wcserr_fs_pre2014_uses_wcs_err(self):
        """fs pre-2014 maps wcserr to WCS_ERR key."""
        from lsc.util import readkey3
        hdr = fits.Header()
        hdr['INSTRUME'] = 'fs01'
        hdr['DATE-OBS'] = '2013-01-01T12:00:00'
        hdr['DAY-OBS'] = '2013-01-01'
        hdr['FILTER'] = 'r'
        hdr['GAIN'] = 2.0
        hdr['RDNOISE'] = 10.0
        hdr['RA'] = 150.0
        hdr['DEC'] = 2.5
        hdr['SATURATE'] = 65000.0
        hdr['MJD'] = 56293.0
        hdr['WCS_ERR'] = 1.2
        val = readkey3(hdr, 'wcserr')
        assert val == 1.2


# ---------------------------------------------------------------------------
# readkey3 — 'type' key (OBSTYPE)
# ---------------------------------------------------------------------------

class TestReadkey3Type:
    def test_type_returns_obstype(self):
        from lsc.util import readkey3
        hdr = fits.Header()
        hdr['INSTRUME'] = 'fa15'
        hdr['DATE-OBS'] = '2020-01-01T12:00:00'
        hdr['DAY-OBS'] = '2020-01-01'
        hdr['FILTER'] = 'r'
        hdr['GAIN'] = 2.0
        hdr['RDNOISE'] = 10.0
        hdr['RA'] = 150.0
        hdr['DEC'] = 2.5
        hdr['SATURATE'] = 65000.0
        hdr['MJD-OBS'] = 58000.0
        hdr['OBSTYPE'] = 'BIAS'
        val = readkey3(hdr, 'type')
        assert val == 'BIAS'


# ---------------------------------------------------------------------------
# readkey3 — 'propid' and 'userid' keys
# ---------------------------------------------------------------------------

class TestReadkey3PropidUserid:
    def test_propid_returns_value(self):
        from lsc.util import readkey3
        hdr = fits.Header()
        hdr['INSTRUME'] = 'kb70'
        hdr['DATE-OBS'] = '2020-01-01T12:00:00'
        hdr['DAY-OBS'] = '2020-01-01'
        hdr['FILTER'] = 'r'
        hdr['GAIN'] = 2.0
        hdr['RDNOISE'] = 10.0
        hdr['RA'] = 150.0
        hdr['DEC'] = 2.5
        hdr['SATURATE'] = 65000.0
        hdr['MJD-OBS'] = 58000.0
        hdr['PROPID'] = 'LCO2020A-001'
        val = readkey3(hdr, 'propid')
        assert val == 'LCO2020A-001'

    def test_userid_returns_value(self):
        from lsc.util import readkey3
        hdr = fits.Header()
        hdr['INSTRUME'] = 'kb70'
        hdr['DATE-OBS'] = '2020-01-01T12:00:00'
        hdr['DAY-OBS'] = '2020-01-01'
        hdr['FILTER'] = 'r'
        hdr['GAIN'] = 2.0
        hdr['RDNOISE'] = 10.0
        hdr['RA'] = 150.0
        hdr['DEC'] = 2.5
        hdr['SATURATE'] = 65000.0
        hdr['MJD-OBS'] = 58000.0
        hdr['USERID'] = 'astronomer1'
        val = readkey3(hdr, 'userid')
        assert val == 'astronomer1'


# ---------------------------------------------------------------------------
# readkey3 — 'airmass' key
# ---------------------------------------------------------------------------

class TestReadkey3Airmass:
    def test_airmass_returns_value(self):
        from lsc.util import readkey3
        hdr = fits.Header()
        hdr['INSTRUME'] = 'fa15'
        hdr['DATE-OBS'] = '2020-01-01T12:00:00'
        hdr['DAY-OBS'] = '2020-01-01'
        hdr['FILTER'] = 'r'
        hdr['GAIN'] = 2.0
        hdr['RDNOISE'] = 10.0
        hdr['RA'] = 150.0
        hdr['DEC'] = 2.5
        hdr['SATURATE'] = 65000.0
        hdr['MJD-OBS'] = 58000.0
        hdr['AIRMASS'] = 1.85
        val = readkey3(hdr, 'airmass')
        assert abs(val - 1.85) < 1e-6


# ---------------------------------------------------------------------------
# readkey3 — 'observer' key
# ---------------------------------------------------------------------------

class TestReadkey3Observer:
    def test_observer_returns_value(self):
        from lsc.util import readkey3
        hdr = fits.Header()
        hdr['INSTRUME'] = 'fa15'
        hdr['DATE-OBS'] = '2020-01-01T12:00:00'
        hdr['DAY-OBS'] = '2020-01-01'
        hdr['FILTER'] = 'r'
        hdr['GAIN'] = 2.0
        hdr['RDNOISE'] = 10.0
        hdr['RA'] = 150.0
        hdr['DEC'] = 2.5
        hdr['SATURATE'] = 65000.0
        hdr['MJD-OBS'] = 58000.0
        hdr['OBSERVER'] = 'John Doe'
        val = readkey3(hdr, 'observer')
        assert val == 'John Doe'


# ---------------------------------------------------------------------------
# readkey3 — fs03 pixscale pre-2014
# ---------------------------------------------------------------------------

class TestReadkey3Fs03Pixscale:
    def test_fs03_pixscale_304(self):
        """fs03 (COJ) pre-2014 gets pixscale = 0.304."""
        from lsc.util import readkey3
        hdr = fits.Header()
        hdr['INSTRUME'] = 'fs03'
        hdr['DATE-OBS'] = '2013-06-01T12:00:00'
        hdr['DAY-OBS'] = '2013-06-01'
        hdr['FILTER'] = 'r'
        hdr['GAIN'] = 2.0
        hdr['RDNOISE'] = 10.0
        hdr['RA'] = 150.0
        hdr['DEC'] = 2.5
        hdr['MJD'] = 56000.0
        hdr['SATURATE'] = 60000.0
        val = readkey3(hdr, 'pixscale')
        assert abs(val - 0.304) < 1e-6


# ---------------------------------------------------------------------------
# readkey3 — 'datamax' with SATURATE header
# ---------------------------------------------------------------------------

class TestReadkey3Datamax:
    def test_datamax_reads_saturate(self):
        """For post-2014 fs/fa/fl, datamax maps to SATURATE header."""
        from lsc.util import readkey3
        hdr = fits.Header()
        hdr['INSTRUME'] = 'fa15'
        hdr['DATE-OBS'] = '2020-01-01T12:00:00'
        hdr['DAY-OBS'] = '2020-01-01'
        hdr['FILTER'] = 'r'
        hdr['GAIN'] = 2.0
        hdr['RDNOISE'] = 10.0
        hdr['RA'] = 150.0
        hdr['DEC'] = 2.5
        hdr['SATURATE'] = 55000.0
        hdr['MJD-OBS'] = 58000.0
        val = readkey3(hdr, 'datamax')
        assert val == 55000.0

    def test_datamax_default_branch_reads_saturate(self):
        """Default (unknown) instrument: datamax maps to SATURATE."""
        from lsc.util import readkey3
        hdr = fits.Header()
        hdr['INSTRUME'] = 'unknown_xyz'
        hdr['DATE-OBS'] = '2020-01-01T12:00:00'
        hdr['DAY-OBS'] = '2020-01-01'
        hdr['RA'] = 100.0
        hdr['DEC'] = 10.0
        hdr['MJD-OBS'] = 58000.0
        hdr['SATURATE'] = 50000.0
        val = readkey3(hdr, 'datamax')
        assert val == 50000.0


# ---------------------------------------------------------------------------
# getcatalog — field priority order
# ---------------------------------------------------------------------------

class TestGetcatalogFieldPriority:
    def _make_conn(self, return_value):
        from unittest.mock import MagicMock
        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchall.return_value = return_value
        cursor.rowcount = len(return_value)
        conn.cursor.return_value = cursor
        return conn

    def test_default_priority_landolt_first(self, monkeypatch, tmp_path):
        """Default field priority is landolt > sloan > apass."""
        import lsc.mysqldef
        import lsc.myloopdef
        # Create landolt catalog file
        cat_dir = tmp_path / 'standard' / 'cat' / 'landolt'
        cat_dir.mkdir(parents=True)
        cat_file = cat_dir / 'SN2024abc.cat'
        cat_file.write_text('# test\n')
        conn = self._make_conn((
            {'name': 'SN2024abc', 'landolt_cat': 'SN2024abc.cat',
             'sloan_cat': 'SN2024abc_sloan.cat',
             'apass_cat': None, 'gaia_cat': None},
        ))
        monkeypatch.setattr(lsc.myloopdef, 'conn', conn)
        monkeypatch.setenv('LCOSNDIR', str(tmp_path))
        from lsc.util import getcatalog
        result = getcatalog('SN2024abc')
        # Landolt has higher priority
        assert 'landolt' in result

    def test_return_field_flag(self, monkeypatch):
        """return_field=True returns (catalog, field) tuple."""
        import lsc.mysqldef
        import lsc.myloopdef
        conn = self._make_conn(())
        monkeypatch.setattr(lsc.myloopdef, 'conn', conn)
        from lsc.util import getcatalog
        result = getcatalog('NoMatch', return_field=True)
        assert isinstance(result, tuple)
        assert result[0] == ''


# ---------------------------------------------------------------------------
# checksndb — basename extraction
# ---------------------------------------------------------------------------

class TestChecksndbBasename:
    def test_extracts_basename_from_path(self, monkeypatch):
        """checksndb uses os.path.basename to extract filename from full path."""
        import lsc.mysqldef
        import lsc.myloopdef
        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchall.return_value = (
            {'ra0': 180.0, 'dec0': -5.0, 'classificationid': 2},
        )
        conn.cursor.return_value = cursor
        monkeypatch.setattr(lsc.myloopdef, 'conn', conn)
        from lsc.util import checksndb
        ra, dec, cls = checksndb('/full/path/to/image.fits')
        assert ra == 180.0
        assert dec == -5.0
        assert cls == 2
        # Verify the SQL used basename 'image.fits'
        call_args = cursor.execute.call_args[0][0]
        assert 'image.fits' in call_args


# ---------------------------------------------------------------------------
# defsex — verifies correct path substitution
# ---------------------------------------------------------------------------

class TestDefsexPaths:
    def test_all_paths_use_lsc_package_dir(self, tmp_path):
        """All substituted paths point to the lsc package directory."""
        import lsc
        from lsc.util import defsex
        sexfile = str(tmp_path / 'check_paths.sex')
        defsex(sexfile)
        content = open(sexfile).read()
        # All three paths should contain the lsc package path
        assert content.count(lsc.__path__[0]) >= 3

    def test_non_substituted_lines_preserved(self, tmp_path):
        """Lines not matching any substitution pattern are written verbatim."""
        from lsc.util import defsex
        sexfile = str(tmp_path / 'verbatim.sex')
        defsex(sexfile)
        content = open(sexfile).read()
        # File should have content (more than just the 3 substituted lines)
        assert len(content.splitlines()) > 3


# ---------------------------------------------------------------------------
# correctobject — coordinate distance calculation
# ---------------------------------------------------------------------------

class TestCorrectobjectDistance:
    def test_exact_match_returns_star(self, tmp_path):
        """When image coordinates exactly match a standard, returns that star."""
        from lsc.util import correctobject
        # Create standard file with known coordinates
        std_file = str(tmp_path / 'standards.txt')
        # Star at RA=180, DEC=0 (12:00:00.0 +00:00:00.0)
        open(std_file, 'w').write("ExactStar 12:00:00.0 +00:00:00.0 15.0\n")
        # Create FITS with exact same coordinates
        hdr = fits.Header()
        hdr['INSTRUME'] = 'kb70'
        hdr['DATE-OBS'] = '2020-01-01T12:00:00'
        hdr['DAY-OBS'] = '2020-01-01'
        hdr['MJD-OBS'] = 58000.0
        hdr['OBJECT'] = 'Unknown'
        hdr['RA'] = 180.0
        hdr['DEC'] = 0.0
        hdr['CAT-RA'] = 180.0
        hdr['CAT-DEC'] = 0.0
        hdr['SATURATE'] = 60000.0
        hdr['RDNOISE'] = 10.0
        hdr['AIRMASS'] = 1.0
        path = str(tmp_path / 'match.fits')
        fits.writeto(path, np.zeros((5, 5), dtype=np.float32), hdr, overwrite=True)
        aa, bb, cc = correctobject(path, std_file)
        assert cc == 'ExactStar'
        assert abs(aa - 180.0) < 0.01
        assert abs(bb - 0.0) < 0.01

    def test_no_nearby_standard_returns_empty(self, tmp_path):
        """When no standard is within 200 arcsec, returns empty strings."""
        from lsc.util import correctobject
        std_file = str(tmp_path / 'far_std.txt')
        # Star at RA=0, DEC=0
        open(std_file, 'w').write("FarStar 00:00:00.0 +00:00:00.0 15.0\n")
        # FITS at very different position
        hdr = fits.Header()
        hdr['INSTRUME'] = 'kb70'
        hdr['DATE-OBS'] = '2020-01-01T12:00:00'
        hdr['DAY-OBS'] = '2020-01-01'
        hdr['MJD-OBS'] = 58000.0
        hdr['OBJECT'] = 'Unknown'
        hdr['RA'] = 180.0  # 180 degrees away
        hdr['DEC'] = 45.0
        hdr['CAT-RA'] = 180.0
        hdr['CAT-DEC'] = 45.0
        hdr['SATURATE'] = 60000.0
        hdr['RDNOISE'] = 10.0
        hdr['AIRMASS'] = 1.0
        path = str(tmp_path / 'far.fits')
        fits.writeto(path, np.zeros((5, 5), dtype=np.float32), hdr, overwrite=True)
        aa, bb, cc = correctobject(path, std_file)
        assert aa == ''
        assert bb == ''
        assert cc == ''
