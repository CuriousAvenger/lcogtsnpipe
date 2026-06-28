"""
Tests for lsc.util — FITS I/O, list utilities, readkey3, cosmic ray rejection,
readstandard, airmass, defswarp, defsex, limmag, and helper functions.
No database, IRAF, or network access required (mocked where needed).
"""
import os
import sys
import re
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, mock_open, call
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

class TestUserinput:
    def test_returns_user_input(self):
        from lsc.util import userinput
        with patch("sys.stdin.isatty", return_value=True), \
             patch("builtins.input", return_value="hello"):
            assert userinput("prompt: ") == "hello"

    def test_passes_prompt_to_input(self):
        from lsc.util import userinput
        captured = []
        with patch("sys.stdin.isatty", return_value=True), \
             patch("builtins.input", side_effect=lambda p: captured.append(p) or ""):
            userinput("enter value: ")
        assert "enter value:" in captured[0]

    def test_returns_empty_string(self):
        from lsc.util import userinput
        with patch("builtins.input", return_value=""):
            assert userinput("") == ""

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
# readhdr
# ---------------------------------------------------------------------------

class TestReadhdr:
    def test_returns_header(self, simple_fits):
        from lsc.util import readhdr
        hdr = readhdr(simple_fits)
        assert hdr is not None

    def test_header_has_exptime(self, simple_fits):
        from lsc.util import readhdr
        hdr = readhdr(simple_fits)
        assert "EXPTIME" in hdr
        assert hdr["EXPTIME"] == 120.0

    def test_header_has_filter(self, simple_fits):
        from lsc.util import readhdr
        hdr = readhdr(simple_fits)
        assert hdr["FILTER"] == "r"

    def test_nonexistent_file_raises(self, tmp_path):
        from lsc.util import readhdr
        with pytest.raises(Exception):
            readhdr(str(tmp_path / "does_not_exist.fits"))

    def test_corrupted_file_raises(self, tmp_path):
        from lsc.util import readhdr
        bad = str(tmp_path / "corrupt.fits")
        open(bad, 'w').write("not a FITS file")
        with pytest.raises(Exception):
            readhdr(bad)


# ---------------------------------------------------------------------------
# readpasswd
# ---------------------------------------------------------------------------

class TestReadpasswd:
    def _write_config(self, tmp_path, content):
        p = tmp_path / "configure"
        p.write_text(content)
        return str(p)

    def test_returns_dict(self, tmp_path):
        from lsc.util import readpasswd
        cfg = self._write_config(tmp_path, "hostname localhost\nmysqluser snuser\n")
        result = readpasswd(cfg)
        assert isinstance(result, dict)

    def test_reads_string_values(self, tmp_path):
        from lsc.util import readpasswd
        cfg = self._write_config(tmp_path, "hostname db.example.com\ndatabase sndb\n")
        result = readpasswd(cfg)
        assert result["hostname"] == "db.example.com"

    def test_reads_numeric_values(self, tmp_path):
        from lsc.util import readpasswd
        cfg = self._write_config(tmp_path, "port 3306\ndatabase sndb\n")
        result = readpasswd(cfg)
        assert result["port"] == 3306

    def test_reads_multiple_keys(self, tmp_path):
        from lsc.util import readpasswd
        cfg = self._write_config(
            tmp_path,
            "hostname localhost\nmysqluser snuser\nmysqlpasswd secret\ndatabase sndb\n"
        )
        result = readpasswd(cfg)
        for key in ("hostname", "mysqluser", "mysqlpasswd", "database"):
            assert key in result

    def test_reads_list_value(self, tmp_path):
        from lsc.util import readpasswd
        cfg = self._write_config(tmp_path, "proposal ['key1','key2']\nusers testuser\n")
        result = readpasswd(cfg)
        assert isinstance(result["proposal"], list)
        assert "key1" in result["proposal"]

    def test_eval_fails_uses_raw_string(self, tmp_path):
        """When eval() fails on a value, the raw string is stored."""
        from lsc.util import readpasswd
        cfg = self._write_config(tmp_path, "key1 some_value_that_cant_eval\nkey2 another\n")
        result = readpasswd(cfg)
        assert result["key1"] == "some_value_that_cant_eval"

    def test_boolean_eval(self, tmp_path):
        from lsc.util import readpasswd
        cfg = self._write_config(tmp_path, "flag True\nother_key value\n")
        result = readpasswd(cfg)
        assert result["flag"] is True

    def test_tuple_eval(self, tmp_path):
        from lsc.util import readpasswd
        cfg = self._write_config(tmp_path, "coords (1,2,3)\nname test\n")
        result = readpasswd(cfg)
        assert result["coords"] == (1, 2, 3)

    def test_dict_eval(self, tmp_path):
        from lsc.util import readpasswd
        cfg = self._write_config(tmp_path, "mapping {'a':1}\nname test\n")
        result = readpasswd(cfg)
        assert result["mapping"] == {'a': 1}


# ---------------------------------------------------------------------------
# ReadAscii2
# ---------------------------------------------------------------------------

class TestReadAscii2:
    def test_reads_two_columns(self, tmp_path):
        from lsc.util import ReadAscii2
        p = tmp_path / "data.txt"
        p.write_text("1.0 2.0\n3.0 4.0\n5.0 6.0\n")
        col1, col2 = ReadAscii2(str(p))
        assert col1 == [1.0, 3.0, 5.0]
        assert col2 == [2.0, 4.0, 6.0]

    def test_skips_comment_lines(self, tmp_path):
        from lsc.util import ReadAscii2
        p = tmp_path / "comments.txt"
        p.write_text("# header comment\n10.0 20.0\n# another comment\n30.0 40.0\n")
        col1, col2 = ReadAscii2(str(p))
        assert col1 == [10.0, 30.0]
        assert col2 == [20.0, 40.0]

    def test_returns_lists(self, tmp_path):
        from lsc.util import ReadAscii2
        p = tmp_path / "single.txt"
        p.write_text("7.5 8.5\n")
        col1, col2 = ReadAscii2(str(p))
        assert isinstance(col1, list) and isinstance(col2, list)

    def test_empty_file_returns_empty_lists(self, tmp_path):
        from lsc.util import ReadAscii2
        p = tmp_path / "empty.txt"
        p.write_text("# only comment\n")
        col1, col2 = ReadAscii2(str(p))
        assert col1 == [] and col2 == []

    def test_negative_values(self, tmp_path):
        from lsc.util import ReadAscii2
        p = tmp_path / "neg.txt"
        p.write_text("-1.5 -2.5\n-3.0 4.0\n")
        col1, col2 = ReadAscii2(str(p))
        assert col1 == [-1.5, -3.0]
        assert col2 == [-2.5, 4.0]

    def test_extra_columns_ignored(self, tmp_path):
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
# readlist
# ---------------------------------------------------------------------------

class TestReadlist:
    def test_reads_single_fits_file(self, simple_fits):
        from lsc.util import readlist
        result = readlist(simple_fits)
        assert simple_fits in result

    def test_text_list_of_files(self, simple_fits, tmp_path):
        from lsc.util import readlist
        list_file = tmp_path / "files.list"
        list_file.write_text(simple_fits + "\n")
        result = readlist(str(list_file))
        assert simple_fits in result

    def test_text_list_ignores_comments(self, simple_fits, tmp_path):
        from lsc.util import readlist
        list_file = tmp_path / "files_with_comments.list"
        list_file.write_text(f"# this is a comment\n{simple_fits}\n")
        result = readlist(str(list_file))
        assert simple_fits in result

    def test_comma_separated(self, simple_fits):
        from lsc.util import readlist
        result = readlist(simple_fits + "," + simple_fits)
        assert len(result) == 2

    def test_nonexistent_file_raises(self, tmp_path):
        from lsc.util import readlist
        with pytest.raises(SystemExit):
            readlist(str(tmp_path / "nosuchfile.notfits"))

    def test_glob_pattern(self, simple_fits):
        from lsc.util import readlist
        directory = os.path.dirname(simple_fits)
        result = readlist(os.path.join(directory, "*.fits"))
        assert any(simple_fits in r for r in result)

    def test_corrupted_fits_in_list_skipped(self, simple_fits, tmp_path):
        from lsc.util import readlist
        bad = tmp_path / "corrupt.fits"
        bad.write_bytes(b'not a valid fits file')
        list_file = tmp_path / "mixed.list"
        list_file.write_text(f"{simple_fits}\n{bad}\n")
        result = readlist(str(list_file))
        assert simple_fits in result

    def test_all_corrupted_fits_exits(self, tmp_path):
        from lsc.util import readlist
        bad = tmp_path / "corrupt.fits"
        bad.write_bytes(b'not a valid fits file')
        list_file = tmp_path / "allbad.list"
        list_file.write_text(f"{bad}\n")
        with pytest.raises(SystemExit):
            readlist(str(list_file))

    def test_glob_no_match_exits(self, tmp_path):
        from lsc.util import readlist
        with pytest.raises(SystemExit):
            readlist(str(tmp_path / "*.nonexistent_extension"))

    def test_multiple_comma_separated(self, simple_fits):
        from lsc.util import readlist
        path_str = ",".join([simple_fits] * 3)
        result = readlist(path_str)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

class TestDelete:
    def test_removes_single_file(self, tmp_path):
        from lsc.util import delete
        f = tmp_path / 'file.fits'
        f.write_text('x')
        delete(str(f))
        assert not f.exists()

    def test_removes_comma_separated(self, tmp_path):
        from lsc.util import delete
        f1 = tmp_path / 'a.fits'
        f2 = tmp_path / 'b.fits'
        f1.write_text('x')
        f2.write_text('x')
        delete(str(f1) + ',' + str(f2))
        assert not f1.exists()
        assert not f2.exists()

    def test_at_sign_reads_list_file(self, tmp_path):
        from lsc.util import delete
        f1 = tmp_path / 'del1.fits'
        f1.write_text('x')
        listfile = tmp_path / 'to_delete.list'
        listfile.write_text(str(f1) + '\n')
        delete('@' + str(listfile))
        assert not f1.exists()

    def test_nonexistent_file_silently_passes(self, tmp_path):
        from lsc.util import delete
        delete(str(tmp_path / 'no_such_file.fits'))

    def test_glob_wildcard(self, tmp_path):
        from lsc.util import delete
        (tmp_path / 'img1.fits').write_text('x')
        (tmp_path / 'img2.fits').write_text('x')
        delete(str(tmp_path / 'img?.fits'))
        assert not (tmp_path / 'img1.fits').exists()
        assert not (tmp_path / 'img2.fits').exists()

    def test_at_list_ignores_comment_lines(self, tmp_path):
        from lsc.util import delete
        f1 = tmp_path / 'delcmmt.fits'
        f1.write_text('x')
        listfile = tmp_path / 'cmt.list'
        listfile.write_text(f'# comment\n{f1}\n')
        delete('@' + str(listfile))
        assert not f1.exists()

    def test_at_sign_skips_blank_lines(self, tmp_path):
        from lsc.util import delete
        f1 = tmp_path / "todel.fits"
        f1.write_text("x")
        listfile = tmp_path / "del.list"
        listfile.write_text(f"\n\n{f1}\n\n")
        delete("@" + str(listfile))
        assert not f1.exists()

    def test_at_sign_skips_whitespace_only_lines(self, tmp_path):
        from lsc.util import delete
        f1 = tmp_path / "delws.fits"
        f1.write_text("x")
        listfile = tmp_path / "ws.list"
        listfile.write_text(f"   \n{f1}\n")
        delete("@" + str(listfile))
        assert not f1.exists()

    def test_os_system_raises_exception_branch(self, tmp_path):
        """Force the except:pass at line 113 by making os.system raise."""
        import lsc.util
        target = tmp_path / "victim.txt"
        target.write_text("x")
        with patch('os.system', side_effect=OSError("forced")):
            lsc.util.delete(str(target))


# ---------------------------------------------------------------------------
# imcopy
# ---------------------------------------------------------------------------

class TestImcopy:
    def test_full_copy_pixel_data(self, simple_fits, tmp_path):
        from lsc.util import imcopy
        dst = str(tmp_path / "copy.fits")
        imcopy(simple_fits, dst)
        orig = fits.getdata(simple_fits)
        copy = fits.getdata(dst)
        np.testing.assert_array_equal(orig, copy)

    def test_full_copy_preserves_header(self, simple_fits, tmp_path):
        from lsc.util import imcopy
        dst = str(tmp_path / "copy_hdr.fits")
        imcopy(simple_fits, dst)
        orig_hdr = fits.getheader(simple_fits)
        copy_hdr = fits.getheader(dst)
        assert orig_hdr["EXPTIME"] == copy_hdr["EXPTIME"]
        assert orig_hdr["FILTER"] == copy_hdr["FILTER"]

    def test_cutout_reduces_shape(self, simple_fits, tmp_path):
        from lsc.util import imcopy
        dst = str(tmp_path / "cutout.fits")
        imcopy(simple_fits, dst, center=(50, 50), cutout_size=(20, 20))
        data = fits.getdata(dst)
        assert data.shape == (20, 20)

    def test_cutout_with_single_size_value(self, simple_fits, tmp_path):
        from lsc.util import imcopy
        dst = str(tmp_path / "square_cutout.fits")
        imcopy(simple_fits, dst, center=(50, 50), cutout_size=20)
        data = fits.getdata(dst)
        assert data.shape == (20, 20)

    def test_no_cutout_copies_full(self, simple_fits, tmp_path):
        from lsc.util import imcopy
        dst = str(tmp_path / "full.fits")
        imcopy(simple_fits, dst, center=None, cutout_size=None)
        data = fits.getdata(dst)
        assert data.shape == (100, 100)

    def test_ext_parameter_defaults_to_zero(self, simple_fits, tmp_path):
        from lsc.util import imcopy
        dst = str(tmp_path / "ext0.fits")
        imcopy(simple_fits, dst, ext=0)
        assert os.path.exists(dst)


# ---------------------------------------------------------------------------
# updateheader
# ---------------------------------------------------------------------------

class TestUpdateheader:
    def test_creates_new_keyword(self, simple_fits, tmp_path):
        from lsc.util import updateheader
        import shutil
        dst = str(tmp_path / "update_test.fits")
        shutil.copy2(simple_fits, dst)
        updateheader(dst, 0, {"TESTKEY": 42})
        hdr = fits.getheader(dst)
        assert hdr["TESTKEY"] == 42

    def test_updates_existing_keyword(self, simple_fits, tmp_path):
        from lsc.util import updateheader
        import shutil
        dst = str(tmp_path / "update_test2.fits")
        shutil.copy2(simple_fits, dst)
        updateheader(dst, 0, {"EXPTIME": 300.0})
        hdr = fits.getheader(dst)
        assert abs(hdr["EXPTIME"] - 300.0) < 1e-5

    def test_multiple_keywords(self, simple_fits, tmp_path):
        from lsc.util import updateheader
        import shutil
        dst = str(tmp_path / "update_test3.fits")
        shutil.copy2(simple_fits, dst)
        updateheader(dst, 0, {"KEY1": "hello", "KEY2": 3.14})
        hdr = fits.getheader(dst)
        assert hdr["KEY1"] == "hello"
        assert abs(hdr["KEY2"] - 3.14) < 1e-5

    def test_tuple_value_with_comment(self, simple_fits, tmp_path):
        from lsc.util import updateheader
        import shutil
        dst = str(tmp_path / "tuple_hdr.fits")
        shutil.copy2(simple_fits, dst)
        updateheader(dst, 0, {"TESTKEY": (99.9, "test comment")})
        hdr = fits.getheader(dst)
        assert abs(hdr["TESTKEY"] - 99.9) < 1e-5
        assert "test comment" in hdr.comments["TESTKEY"]

    def test_invalid_dimension_prints_error(self, simple_fits, tmp_path, capsys):
        from lsc.util import updateheader
        import shutil
        dst = str(tmp_path / "bad_dim.fits")
        shutil.copy2(simple_fits, dst)
        updateheader(dst, 5, {"KEY": 1})
        out = capsys.readouterr().out
        assert 'not updated' in out

    def test_empty_headerdict(self, simple_fits, tmp_path):
        from lsc.util import updateheader
        import shutil
        dst = str(tmp_path / "empty_hdr.fits")
        shutil.copy2(simple_fits, dst)
        updateheader(dst, 0, {})
        hdr = fits.getheader(dst)
        assert hdr is not None

    def test_nonexistent_file_does_not_raise(self, capsys):
        from lsc.util import updateheader
        updateheader('/nonexistent/file.fits', 0, {'KEY': 99})
        out = capsys.readouterr().out
        assert 'not updated' in out or 'file.fits' in out


# ---------------------------------------------------------------------------
# pval and residual
# ---------------------------------------------------------------------------

class TestPval:
    def test_basic(self):
        from lsc.util import pval
        assert abs(pval(3.0, [1.0, 2.0]) - 7.0) < 1e-9

    def test_zero_slope(self):
        from lsc.util import pval
        assert abs(pval(100.0, [5.0, 0.0]) - 5.0) < 1e-9

    def test_negative_x(self):
        from lsc.util import pval
        assert abs(pval(-2.0, [0.0, 3.0]) - (-6.0)) < 1e-9

    def test_zero_intercept(self):
        from lsc.util import pval
        assert abs(pval(4.0, [0.0, 2.5]) - 10.0) < 1e-9

    def test_numpy_array_input(self):
        from lsc.util import pval
        x = np.array([1.0, 2.0, 3.0])
        result = pval(x, [1.0, 2.0])
        expected = np.array([3.0, 5.0, 7.0])
        np.testing.assert_allclose(result, expected)

    def test_zero_coefficients(self):
        from lsc.util import pval
        result = pval(5.0, [0.0, 0.0])
        assert result == 0.0


class TestResidual:
    def test_two_coefficients_returns_last_step(self):
        from lsc.util import residual
        err = residual([1.0, 2.0], 5.0, 3.0)
        assert abs(err - (-1.0)) < 1e-9

    def test_single_coefficient(self):
        from lsc.util import residual
        err = residual([3.0], 10.0, 4.0)
        assert abs(err - 7.0) < 1e-9

    def test_perfect_fit(self):
        from lsc.util import residual
        err = residual([0.0, 2.0], 6.0, 3.0)
        assert abs(err) < 1e-9

    def test_three_coefficients(self):
        from lsc.util import residual
        err = residual([1.0, 2.0, 3.0], 10.0, 2.0)
        assert abs(err - (-2.0)) < 1e-9

    def test_numpy_arrays(self):
        from lsc.util import residual
        y = np.array([5.0, 10.0])
        x = np.array([1.0, 2.0])
        err = residual([0.0, 2.0], y, x)
        expected = np.array([3.0, 6.0])
        np.testing.assert_allclose(err, expected)

    def test_zero_x(self):
        from lsc.util import residual
        err = residual([3.0, 5.0], 10.0, 0.0)
        assert abs(err - 10.0) < 1e-9


# ---------------------------------------------------------------------------
# writeinthelog
# ---------------------------------------------------------------------------

class TestWriteinthelog:
    def test_creates_file(self, tmp_path):
        from lsc.util import writeinthelog
        logfile = str(tmp_path / 'test.log')
        writeinthelog('hello world\n', logfile)
        assert os.path.exists(logfile)

    def test_appends_content(self, tmp_path):
        from lsc.util import writeinthelog
        logfile = str(tmp_path / 'append.log')
        writeinthelog('line1\n', logfile)
        writeinthelog('line2\n', logfile)
        content = open(logfile).read()
        assert 'line1' in content
        assert 'line2' in content

    def test_empty_text(self, tmp_path):
        from lsc.util import writeinthelog
        logfile = str(tmp_path / 'empty.log')
        writeinthelog('', logfile)
        assert open(logfile).read() == ''

    def test_unicode_content(self, tmp_path):
        from lsc.util import writeinthelog
        logfile = str(tmp_path / 'unicode.log')
        writeinthelog('stars: ★ ☆\n', logfile)
        content = open(logfile, encoding='utf-8').read()
        assert '★' in content

    def test_tab_characters(self, tmp_path):
        from lsc.util import writeinthelog
        logfile = str(tmp_path / 'tabs.log')
        writeinthelog('col1\tcol2\tcol3\n', logfile)
        content = open(logfile).read()
        assert '\t' in content


# ---------------------------------------------------------------------------
# repstringinfile
# ---------------------------------------------------------------------------

class TestRepstringinfile:
    def test_replaces_string(self, tmp_path):
        from lsc.util import repstringinfile
        fin = str(tmp_path / 'input.txt')
        fout = str(tmp_path / 'output.txt')
        open(fin, 'w').write('hello world\nfoo bar\n')
        repstringinfile(fin, fout, 'hello', 'goodbye')
        content = open(fout).read()
        assert 'goodbye world' in content
        assert 'hello' not in content

    def test_no_match_unchanged(self, tmp_path):
        from lsc.util import repstringinfile
        fin = str(tmp_path / 'input.txt')
        fout = str(tmp_path / 'output.txt')
        open(fin, 'w').write('hello world\n')
        repstringinfile(fin, fout, 'nonexistent', 'replacement')
        assert open(fout).read() == 'hello world\n'

    def test_multiple_occurrences(self, tmp_path):
        from lsc.util import repstringinfile
        fin = str(tmp_path / 'input.txt')
        fout = str(tmp_path / 'output.txt')
        open(fin, 'w').write('foo bar foo baz foo\n')
        repstringinfile(fin, fout, 'foo', 'qux')
        content = open(fout).read()
        assert content.count('qux') == 3
        assert 'foo' not in content

    def test_empty_file(self, tmp_path):
        from lsc.util import repstringinfile
        fin = str(tmp_path / 'input.txt')
        fout = str(tmp_path / 'output.txt')
        open(fin, 'w').write('')
        repstringinfile(fin, fout, 'a', 'b')
        assert open(fout).read() == ''

    def test_regex_metacharacters_in_string(self, tmp_path):
        from lsc.util import repstringinfile
        fin = str(tmp_path / 'in.txt')
        fout = str(tmp_path / 'out.txt')
        open(fin, 'w').write('hello.world\n')
        repstringinfile(fin, fout, 'hello.world', 'REPLACED')
        content = open(fout).read()
        assert 'REPLACED' in content

    def test_same_input_output_file(self, tmp_path):
        from lsc.util import repstringinfile
        f = str(tmp_path / 'inplace.txt')
        open(f, 'w').write('original text\n')
        repstringinfile(f, f, 'original', 'modified')
        content = open(f).read()
        assert 'modified text' in content


# ---------------------------------------------------------------------------
# readkey3 — tests spanning many instrument branches
# ---------------------------------------------------------------------------

def _make_fits_with_header(tmp_path, filename, instrume, date_obs, extra=None):
    """Create a minimal FITS file with the given instrument header."""
    hdr = fits.Header()
    hdr['INSTRUME'] = instrume
    hdr['DATE-OBS'] = date_obs
    hdr['DAY-OBS'] = date_obs.split('T')[0]
    hdr['OBJECT'] = 'SN2024test'
    hdr['EXPTIME'] = 60.0
    hdr['FILTER'] = 'g'
    hdr['GAIN'] = 1.5
    hdr['RDNOISE'] = 10.0
    hdr['RDNOIS'] = 10.0
    hdr['RA'] = 150.0
    hdr['DEC'] = 2.5
    hdr['CAT-RA'] = 150.0
    hdr['CAT-DEC'] = 2.5
    hdr['AIRMASS'] = 1.2
    hdr['OBSTYPE'] = 'EXPOSE'
    hdr['SATURATE'] = 65000.0
    hdr['MJD-OBS'] = 58000.0
    hdr['MJD'] = 58000.0
    hdr['WCSERR'] = 0
    hdr['WCS_ERR'] = 0.0
    hdr['TELESCOP'] = 'tel01'
    hdr['PROPID'] = 'KEY2014A-001'
    hdr['USERID'] = 'testuser'
    hdr['OBSERVER'] = 'Test Observer'
    if extra:
        for k, v in extra.items():
            hdr[k] = v
    data = np.zeros((10, 10), dtype=np.float32)
    fpath = str(tmp_path / filename)
    fits.writeto(fpath, data, hdr, overwrite=True)
    return fpath


class TestReadkey3Basic:
    def test_exptime_key(self, simple_fits):
        from lsc.util import readhdr, readkey3
        hdr = readhdr(simple_fits)
        assert abs(float(readkey3(hdr, "exptime")) - 120.0) < 1e-5

    def test_gain_key(self, simple_fits):
        from lsc.util import readhdr, readkey3
        hdr = readhdr(simple_fits)
        assert abs(float(readkey3(hdr, "gain")) - 2.0) < 1e-5

    def test_filter_key(self, simple_fits):
        from lsc.util import readhdr, readkey3
        hdr = readhdr(simple_fits)
        assert readkey3(hdr, "filter") == "r"

    def test_object_key(self, simple_fits):
        from lsc.util import readhdr, readkey3
        hdr = readhdr(simple_fits)
        assert "SN2024abc" in readkey3(hdr, "object")

    def test_unknown_key_returns_empty(self, simple_fits):
        from lsc.util import readhdr, readkey3
        hdr = readhdr(simple_fits)
        assert readkey3(hdr, "NONEXISTENT_KEY") == ""


class TestReadkey3Instruments:
    """Tests for instrument-specific branches (kb, ep, fs, fa, em, default)."""

    def test_kb_exptime(self, tmp_path):
        from lsc.util import readhdr, readkey3
        f = _make_fits_with_header(tmp_path, 'kb.fits', 'kb70', '2020-01-01T12:00:00')
        hdr = readhdr(f)
        assert abs(float(readkey3(hdr, 'exptime')) - 60.0) < 1e-5

    def test_kb_jd_adds_half(self, tmp_path):
        from lsc.util import readhdr, readkey3
        f = _make_fits_with_header(tmp_path, 'kb_jd.fits', 'kb70', '2020-01-01T12:00:00')
        hdr = readhdr(f)
        assert abs(float(readkey3(hdr, 'JD')) - 58000.5) < 1e-5

    def test_ep_exptime(self, tmp_path):
        from lsc.util import readhdr, readkey3
        f = _make_fits_with_header(tmp_path, 'ep.fits', 'ep01', '2020-06-01T00:00:00')
        hdr = readhdr(f)
        assert abs(float(readkey3(hdr, 'exptime')) - 60.0) < 1e-5

    def test_fs_post2014_mjd(self, tmp_path):
        from lsc.util import readhdr, readkey3
        f = _make_fits_with_header(tmp_path, 'fs_post.fits', 'fs01', '2020-03-01T00:00:00')
        hdr = readhdr(f)
        assert abs(float(readkey3(hdr, 'mjd')) - 58000.0) < 1e-5

    def test_fs_pre2014_datamax(self, tmp_path):
        from lsc.util import readhdr, readkey3
        f = _make_fits_with_header(tmp_path, 'fs_pre.fits', 'fs01', '2013-01-01T00:00:00')
        hdr = readhdr(f)
        assert readkey3(hdr, 'datamax') == 60000.0

    def test_fs03_pixscale_304(self, tmp_path):
        from lsc.util import readhdr, readkey3
        f = _make_fits_with_header(tmp_path, 'fs03.fits', 'fs03', '2013-06-01T12:00:00',
                                   extra={'MJD': 56000.0})
        hdr = readhdr(f)
        assert abs(readkey3(hdr, 'pixscale') - 0.304) < 1e-6

    def test_em_instrument_post2014(self, tmp_path):
        from lsc.util import readhdr, readkey3
        f = _make_fits_with_header(tmp_path, 'em.fits', 'em01', '2020-01-01T12:00:00',
                                   extra={'EXPTIME': 30.0, 'RDNOISE': 8.0})
        hdr = readhdr(f)
        assert readkey3(hdr, 'exptime') == 30.0
        assert readkey3(hdr, 'ron') == 8.0

    def test_unknown_instrument_fallback(self, tmp_path):
        from lsc.util import readhdr, readkey3
        f = _make_fits_with_header(tmp_path, 'def.fits', 'unknown_xyz', '2020-01-01T00:00:00')
        hdr = readhdr(f)
        assert readkey3(hdr, 'mjd') == 58000.0
        assert readkey3(hdr, 'NONEXISTENT') == ''

    def test_ftn_becomes_2m0_01(self, tmp_path):
        from lsc.util import readhdr, readkey3
        f = _make_fits_with_header(tmp_path, 'ftn.fits', 'fl16', '2020-01-01T00:00:00',
                                   extra={'TELESCOP': 'ftn'})
        hdr = readhdr(f)
        assert readkey3(hdr, 'telescop') == '2m0-01'

    def test_fts_becomes_2m0_02(self, tmp_path):
        from lsc.util import readhdr, readkey3
        f = _make_fits_with_header(tmp_path, 'fts.fits', 'fl16', '2020-01-01T00:00:00',
                                   extra={'TELESCOP': 'fts'})
        hdr = readhdr(f)
        assert readkey3(hdr, 'telescop') == '2m0-02'

    def test_date_obs_strips_time(self, tmp_path):
        from lsc.util import readhdr, readkey3
        f = _make_fits_with_header(tmp_path, 'dateobs.fits', 'fa15', '2020-03-15T08:30:00')
        hdr = readhdr(f)
        assert readkey3(hdr, 'date-obs') == '20200315'

    def test_filter_air_uses_filter2(self, tmp_path):
        from lsc.util import readhdr, readkey3
        f = _make_fits_with_header(tmp_path, 'airfilter.fits', 'fa15', '2022-01-01T00:00:00',
                                   extra={'FILTER': 'air', 'FILTER2': 'rp'})
        hdr = readhdr(f)
        assert readkey3(hdr, 'filter') == 'rp'

    def test_ra_colon_format_hourangle(self, tmp_path):
        from lsc.util import readhdr, readkey3
        f = _make_fits_with_header(tmp_path, 'racolon.fits', 'kb01', '2022-01-01T00:00:00',
                                   extra={'RA': '10:00:00.0', 'DEC': '02:00:00.0'})
        hdr = readhdr(f)
        ra = readkey3(hdr, 'RA')
        assert abs(ra - 150.0) < 0.01

    def test_instrume_lowercased(self, tmp_path):
        from lsc.util import readhdr, readkey3
        f = _make_fits_with_header(tmp_path, 'instrume.fits', 'FA15', '2020-01-01T00:00:00')
        hdr = readhdr(f)
        assert readkey3(hdr, 'instrume') == 'fa15'


# ---------------------------------------------------------------------------
# readstandard
# ---------------------------------------------------------------------------

class TestReadstandard:
    def test_returns_four_arrays(self):
        import lsc
        from lsc.util import readstandard
        stdfile = os.path.join(lsc.__path__[0], "standard", "stdlist", "standardlist.txt")
        star, ra, dec, mag = readstandard(stdfile)
        assert len(star) > 0
        assert len(star) == len(ra) == len(dec) == len(mag)

    def test_ra_in_degrees(self):
        import lsc
        from lsc.util import readstandard
        stdfile = os.path.join(lsc.__path__[0], "standard", "stdlist", "standardlist.txt")
        star, ra, dec, mag = readstandard(stdfile)
        assert all(0 <= float(r) <= 360 for r in ra)

    def test_custom_file(self, tmp_path):
        from lsc.util import readstandard
        p = tmp_path / "custom.txt"
        p.write_text("Star1 12:00:00.0 +02:00:00.0 15.3\n")
        star, ra, dec, mag = readstandard(str(p))
        assert len(star) == 1
        assert abs(float(ra[0]) - 180.0) < 0.01

    def test_negative_declination(self, tmp_path):
        from lsc.util import readstandard
        p = tmp_path / "negdec.txt"
        p.write_text("Star1 12:00:00.0 -30:30:00.0 16.0\n")
        star, ra, dec, mag = readstandard(str(p))
        assert abs(float(dec[0]) - (-30.5)) < 0.01

    def test_missing_magnitude_defaults_to_999(self, tmp_path):
        from lsc.util import readstandard
        p = tmp_path / "nomag.txt"
        p.write_text("Star3 12:00:00.0 +02:00:00.0\n")
        star, ra, dec, mag = readstandard(str(p))
        assert mag[0] == 999

    def test_comment_lines_skipped(self, tmp_path):
        from lsc.util import readstandard
        p = tmp_path / "comments.txt"
        content = "# Header\nStar1 12:00:00.0 +02:00:00.0 15.0\n# Comment\nStar2 13:00:00.0 -03:00:00.0 16.0\n"
        p.write_text(content)
        star, ra, dec, mag = readstandard(str(p))
        assert len(star) == 2

    def test_absolute_nonexistent_path_raises(self):
        from lsc.util import readstandard
        with pytest.raises((FileNotFoundError, IOError)):
            readstandard('/tmp/__nonexistent_standard_file_xyz__.txt')


# ---------------------------------------------------------------------------
# readspectrum
# ---------------------------------------------------------------------------

class TestReadspectrum:
    def _make_1d_spectrum(self, tmp_path, name, crpix=1, crval=4000.0, cdelt=2.0, n=100):
        hdr = fits.Header()
        hdr['NAXIS1'] = n
        hdr['CRPIX1'] = crpix
        hdr['CRVAL1'] = crval
        hdr['CDELT1'] = cdelt
        flux = np.ones(n, dtype=np.float32)
        fpath = str(tmp_path / name)
        fits.writeto(fpath, flux, hdr, overwrite=True)
        return fpath

    def test_returns_wavelength_and_flux(self, tmp_path):
        from lsc.util import readspectrum
        f = self._make_1d_spectrum(tmp_path, 'spec1d.fits')
        lam, fl = readspectrum(f)
        assert len(lam) == 100
        assert len(fl) == 100

    def test_wavelength_starts_at_crval(self, tmp_path):
        from lsc.util import readspectrum
        f = self._make_1d_spectrum(tmp_path, 'spec_wl.fits', crpix=1, crval=3500.0, cdelt=3.0)
        lam, fl = readspectrum(f)
        assert abs(lam[0] - 3500.0) < 0.01

    def test_cd1_1_fallback(self, tmp_path):
        from lsc.util import readspectrum
        hdr = fits.Header()
        hdr['NAXIS1'] = 50
        hdr['CRPIX1'] = 1
        hdr['CRVAL1'] = 5000.0
        hdr['CD1_1'] = 3.0
        fpath = str(tmp_path / 'spec_cd1_1.fits')
        fits.writeto(fpath, np.ones(50, dtype=np.float32), hdr, overwrite=True)
        lam, fl = readspectrum(fpath)
        assert len(lam) == 50
        assert np.allclose(np.diff(lam), 3.0, atol=1e-6)

    def test_2d_data_takes_first_column(self, tmp_path):
        from lsc.util import readspectrum
        data = np.ones((50, 3), dtype=np.float32)
        data[:, 0] = np.arange(50)
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

    def test_3d_data_takes_first_slice(self, tmp_path):
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

    def test_no_wcs_keywords_returns_empty_lam(self, tmp_path):
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
# defsex
# ---------------------------------------------------------------------------

class TestDefsex:
    def test_creates_file(self, tmp_path):
        from lsc.util import defsex
        sexfile = str(tmp_path / 'test.sex')
        result = defsex(sexfile)
        assert os.path.exists(sexfile)
        assert result == sexfile

    def test_parameters_name_replaced(self, tmp_path):
        import lsc
        from lsc.util import defsex
        sexfile = str(tmp_path / 'params.sex')
        defsex(sexfile)
        content = open(sexfile).read()
        assert lsc.__path__[0] + '/standard/sex/default.param' in content

    def test_filter_name_replaced(self, tmp_path):
        import lsc
        from lsc.util import defsex
        sexfile = str(tmp_path / 'filter.sex')
        defsex(sexfile)
        content = open(sexfile).read()
        assert lsc.__path__[0] + '/standard/sex/default.conv' in content

    def test_starnnw_name_replaced(self, tmp_path):
        import lsc
        from lsc.util import defsex
        sexfile = str(tmp_path / 'nnw.sex')
        defsex(sexfile)
        content = open(sexfile).read()
        assert lsc.__path__[0] + '/standard/sex/default.nnw' in content


# ---------------------------------------------------------------------------
# defswarp
# ---------------------------------------------------------------------------

class TestDefswarp:
    def test_creates_file(self, tmp_path):
        from lsc.util import defswarp
        sfile = str(tmp_path / 'test.swarp')
        result = defswarp(sfile, 'out.fits', 'median')
        assert os.path.exists(sfile)
        assert result == sfile

    def test_imageout_name_replaced(self, tmp_path):
        from lsc.util import defswarp
        sfile = str(tmp_path / 'out.swarp')
        defswarp(sfile, 'myimage.fits', 'average')
        content = open(sfile).read()
        assert 'myimage.fits' in content

    def test_combine_type_median(self, tmp_path):
        from lsc.util import defswarp
        sfile = str(tmp_path / 'median.swarp')
        defswarp(sfile, 'out.fits', 'median')
        assert 'MEDIAN' in open(sfile).read()

    def test_combine_type_average(self, tmp_path):
        from lsc.util import defswarp
        sfile = str(tmp_path / 'avg.swarp')
        defswarp(sfile, 'out.fits', 'Average')
        assert 'AVERAGE' in open(sfile).read()

    def test_combine_type_sum(self, tmp_path):
        from lsc.util import defswarp
        sfile = str(tmp_path / 'sum.swarp')
        defswarp(sfile, 'out.fits', 'sum')
        assert 'SUM' in open(sfile).read()

    def test_gain_replaces_default(self, tmp_path):
        from lsc.util import defswarp
        sfile = str(tmp_path / 'gain.swarp')
        defswarp(sfile, 'out.fits', 'median', gain=2.5)
        assert '2.5' in open(sfile).read()

    def test_without_gain_ron(self, tmp_path):
        from lsc.util import defswarp
        out = tmp_path / 'test.swarp'
        result = defswarp(str(out), 'out.fits', 'median', gain='', ron='')
        assert result == str(out)
        assert out.exists()

    def test_pixelscale_written(self, tmp_path):
        from lsc.util import defswarp
        sfile = str(tmp_path / 'pscale.swarp')
        defswarp(sfile, 'out.fits', 'median', pixelscale=0.389)
        assert '0.389' in open(sfile).read()

    def test_center_written(self, tmp_path):
        from lsc.util import defswarp
        sfile = str(tmp_path / 'center.swarp')
        defswarp(sfile, 'out.fits', 'median', _ra=180.0, _dec=2.5)
        content = open(sfile).read()
        assert '180.0' in content
        assert '2.5' in content

    def test_weightout_name_derived(self, tmp_path):
        from lsc.util import defswarp
        sfile = str(tmp_path / 'weight.swarp')
        defswarp(sfile, 'science.fits', 'median')
        assert 'science.weight.fits' in open(sfile).read()


# ---------------------------------------------------------------------------
# limmag
# ---------------------------------------------------------------------------

class TestLimmag:
    def _make_fits_with_phot(self, tmp_path, name, zp=25.0, mbkg=500.0, fwhm=3.0, **kw):
        hdr = fits.Header()
        hdr['INSTRUME'] = kw.get('instrume', 'kb70')
        hdr['DATE-OBS'] = '2020-01-01T00:00:00'
        hdr['MJD-OBS'] = 58000.0
        hdr['DAY-OBS'] = '2020-01-01'
        hdr['GAIN'] = kw.get('gain', 2.0)
        hdr['RDNOISE'] = 10.0
        hdr['SATURATE'] = 60000.0
        hdr['EXPTIME'] = kw.get('exptime', 120.0)
        hdr['FILTER'] = 'r'
        hdr['RA'] = 150.0
        hdr['DEC'] = 2.5
        hdr['PHOTZP'] = zp
        hdr['MBKG'] = mbkg
        hdr['PSF_FWHM'] = fwhm
        for k, v in kw.items():
            if k not in ('instrume', 'gain', 'exptime'):
                hdr[k] = v
        fpath = str(tmp_path / name)
        fits.writeto(fpath, np.ones((10, 10), dtype=np.float32), hdr, overwrite=True)
        return fpath

    def test_returns_float(self, tmp_path):
        from lsc.util import limmag
        f = self._make_fits_with_phot(tmp_path, 'phot.fits')
        result = limmag(f)
        assert isinstance(result, (int, float))

    def test_missing_photzp_returns_empty(self, tmp_path):
        from lsc.util import limmag
        f = self._make_fits_with_phot(tmp_path, 'nozp.fits')
        with fits.open(f, mode='update') as hdul:
            del hdul[0].header['PHOTZP']
        assert limmag(f) == ''

    def test_missing_fwhm_returns_empty(self, tmp_path):
        from lsc.util import limmag
        f = self._make_fits_with_phot(tmp_path, 'nofwhm.fits')
        with fits.open(f, mode='update') as hdul:
            del hdul[0].header['PSF_FWHM']
        assert limmag(f) == ''

    def test_valid_parameters_returns_reasonable_mag(self, tmp_path):
        from lsc.util import limmag
        f = self._make_fits_with_phot(tmp_path, 'good.fits')
        result = limmag(f)
        assert isinstance(result, float)
        assert 15.0 < result < 30.0

    def test_negative_mbkg_clamped(self, tmp_path):
        from lsc.util import limmag
        f = self._make_fits_with_phot(tmp_path, 'negbkg.fits', mbkg=-10.0)
        result = limmag(f)
        assert result is not None


# ---------------------------------------------------------------------------
# Docosmic (mocked astroscrappy)
# ---------------------------------------------------------------------------

class TestDocosmic:
    def test_returns_three_filenames(self, simple_fits, tmp_path, monkeypatch):
        mock_ast = MagicMock()
        mock_ast.detect_cosmics.return_value = (
            np.zeros((100, 100), dtype=bool),
            np.zeros((100, 100), dtype=np.float32),
        )
        monkeypatch.setitem(sys.modules, 'astroscrappy', mock_ast)
        from lsc.util import Docosmic
        monkeypatch.chdir(tmp_path)
        out, outmask, outsat = Docosmic(simple_fits)
        assert out.endswith('.clean.fits')
        assert outmask.endswith('.mask.fits')
        assert outsat.endswith('.sat.fits')

    def test_creates_output_files(self, simple_fits, tmp_path, monkeypatch):
        mock_ast = MagicMock()
        mock_ast.detect_cosmics.return_value = (
            np.zeros((100, 100), dtype=bool),
            np.zeros((100, 100), dtype=np.float32),
        )
        monkeypatch.setitem(sys.modules, 'astroscrappy', mock_ast)
        from lsc.util import Docosmic
        monkeypatch.chdir(tmp_path)
        out, outmask, outsat = Docosmic(simple_fits)
        assert os.path.exists(out)
        assert os.path.exists(outmask)
        assert os.path.exists(outsat)

    def test_temp_in_filename_produces_float_mask(self, tmp_path, monkeypatch):
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
        path = str(tmp_path / 'temp_image.fits')
        fits.writeto(path, np.ones((20, 20), dtype=np.float32), hdr, overwrite=True)
        monkeypatch.chdir(tmp_path)
        out, outmask, outsat = Docosmic(path)
        mask_data = fits.getdata(outmask)
        assert np.issubdtype(mask_data.dtype, np.floating)


# ---------------------------------------------------------------------------
# Docosmic_old
# ---------------------------------------------------------------------------

class TestDocosmic_old:
    def test_returns_three_filenames(self, simple_fits, tmp_path, monkeypatch):
        from lsc.util import Docosmic_old
        monkeypatch.chdir(tmp_path)
        out, outmask, outsat = Docosmic_old(simple_fits)
        assert out.endswith('.clean.fits')
        assert outmask.endswith('.mask.fits')
        assert outsat.endswith('.sat.fits')

    def test_creates_output_files(self, simple_fits, tmp_path, monkeypatch):
        from lsc.util import Docosmic_old
        monkeypatch.chdir(tmp_path)
        out, outmask, outsat = Docosmic_old(simple_fits)
        assert os.path.exists(out)
        assert os.path.exists(outmask)
        assert os.path.exists(outsat)


# ---------------------------------------------------------------------------
# name_duplicate
# ---------------------------------------------------------------------------

class TestNameDuplicate:
    def _make_fits(self, tmp_path, name, date_obs):
        hdr = fits.Header()
        hdr['DATE-OBS'] = date_obs
        hdr['INSTRUME'] = 'kb70'
        hdr['MJD-OBS'] = 58000.0
        hdr['DAY-OBS'] = date_obs.split('T')[0]
        hdr['OBJECT'] = 'SN2020abc'
        hdr['RA'] = 180.0
        hdr['DEC'] = 2.0
        hdr['SATURATE'] = 60000.0
        hdr['RDNOISE'] = 10.0
        fpath = str(tmp_path / name)
        fits.writeto(fpath, np.zeros((5, 5), dtype=np.float32), hdr, overwrite=True)
        return fpath

    def test_no_existing_file_returns_name_1(self, tmp_path, monkeypatch):
        from lsc.util import name_duplicate
        monkeypatch.chdir(tmp_path)
        base = str(tmp_path / 'SN2020abc_g')
        src = self._make_fits(tmp_path, 'source.fits', '2020-01-01T00:00:00')
        result = name_duplicate(src, base, '_band')
        assert result.endswith('_1_band.fits')

    def test_same_date_obs_reuses_existing(self, tmp_path, monkeypatch):
        from lsc.util import name_duplicate
        monkeypatch.chdir(tmp_path)
        src = self._make_fits(tmp_path, 'source2.fits', '2020-02-01T12:00:00')
        existing = str(tmp_path / 'existing_1_band.fits')
        self._make_fits(tmp_path, 'existing_1_band.fits', '2020-02-01T12:00:00')
        base = str(tmp_path / 'existing')
        result = name_duplicate(src, base, '_band')
        assert result == existing


# ---------------------------------------------------------------------------
# correctobject
# ---------------------------------------------------------------------------

class TestCorrectobject:
    def test_exact_match_returns_star(self, tmp_path):
        from lsc.util import correctobject
        std_file = str(tmp_path / 'standards.txt')
        open(std_file, 'w').write("ExactStar 12:00:00.0 +00:00:00.0 15.0\n")
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

    def test_no_nearby_standard_returns_empty(self, tmp_path):
        from lsc.util import correctobject
        std_file = str(tmp_path / 'far_std.txt')
        open(std_file, 'w').write("FarStar 00:00:00.0 +00:00:00.0 15.0\n")
        hdr = fits.Header()
        hdr['INSTRUME'] = 'kb70'
        hdr['DATE-OBS'] = '2020-01-01T12:00:00'
        hdr['DAY-OBS'] = '2020-01-01'
        hdr['MJD-OBS'] = 58000.0
        hdr['OBJECT'] = 'Unknown'
        hdr['RA'] = 180.0
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
        assert cc == ''


# ---------------------------------------------------------------------------
# checksndb (mocked)
# ---------------------------------------------------------------------------

class TestChecksndb:
    def test_returns_triple_from_db(self, monkeypatch):
        import lsc.mysqldef
        import lsc.myloopdef
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = (
            {'ra0': 150.0, 'dec0': 2.0, 'classificationid': 1},
        )
        monkeypatch.setattr(lsc.myloopdef, 'conn', mock_conn)
        from lsc.util import checksndb
        ra, dec, cls = checksndb('test.fits')
        assert ra == 150.0
        assert dec == 2.0
        assert cls == 1

    def test_returns_empty_triple_when_no_match(self, monkeypatch):
        import lsc.mysqldef
        import lsc.myloopdef
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = ()
        monkeypatch.setattr(lsc.myloopdef, 'conn', mock_conn)
        from lsc.util import checksndb
        ra, dec, cls = checksndb('nonexistent.fits')
        assert ra == '' and dec == '' and cls == ''


# ---------------------------------------------------------------------------
# getcatalog (mocked)
# ---------------------------------------------------------------------------

class TestGetcatalog:
    def _make_conn(self, return_value):
        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchall.return_value = return_value
        cursor.rowcount = len(return_value)
        conn.cursor.return_value = cursor
        return conn

    def test_returns_empty_when_no_db_match(self, monkeypatch):
        import lsc.mysqldef
        import lsc.myloopdef
        conn = self._make_conn(())
        monkeypatch.setattr(lsc.myloopdef, 'conn', conn)
        from lsc.util import getcatalog
        result = getcatalog('SN2024abc')
        assert result == ''

    def test_return_field_true(self, monkeypatch):
        import lsc.mysqldef
        import lsc.myloopdef
        conn = self._make_conn(())
        monkeypatch.setattr(lsc.myloopdef, 'conn', conn)
        from lsc.util import getcatalog
        result = getcatalog('SN2024abc', field='sloan', return_field=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
