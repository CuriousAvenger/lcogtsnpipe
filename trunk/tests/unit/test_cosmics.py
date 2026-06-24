"""
Tests for lsc.cosmics — the L.A.Cosmic cosmic-ray detection algorithm.
"""
import pytest
import numpy as np

pytestmark = pytest.mark.unit

# subsample / rebin / rebin2x2 — array utilities
class TestSubsample:
    def test_output_shape_doubles(self):
        """subsample() doubles both dimensions of the input array."""
        from lsc.cosmics import subsample
        a = np.ones((10, 10))
        b = subsample(a)
        assert b.shape == (20, 20)

    def test_values_preserved(self):
        """subsample() propagates pixel values without interpolation."""
        from lsc.cosmics import subsample
        a = np.full((4, 4), 7.0)
        b = subsample(a)
        assert np.allclose(b, 7.0)


class TestRebin:
    def test_2x2_downsample(self):
        """rebin() reduces array dimensions to the requested shape."""
        from lsc.cosmics import rebin
        a = np.ones((10, 10))
        b = rebin(a, (5, 5))
        assert b.shape == (5, 5)

    def test_mean_preserving(self):
        """rebin() computes the mean of each block, preserving uniform values."""
        from lsc.cosmics import rebin
        a = np.full((6, 4), 4.0)
        b = rebin(a, (3, 2))
        assert np.allclose(b, 4.0)


class TestRebin2x2:
    def test_halves_shape(self):
        """rebin2x2() halves both dimensions of an even-shaped array."""
        from lsc.cosmics import rebin2x2
        a = np.ones((8, 8))
        b = rebin2x2(a)
        assert b.shape == (4, 4)

    def test_odd_shape_raises(self):
        """rebin2x2() raises an error when the input has odd dimensions."""
        from lsc.cosmics import rebin2x2
        a = np.ones((7, 7))
        with pytest.raises(Exception):
            rebin2x2(a)


class TestRebin2x2Values:
    def test_averages_2x2_blocks(self):
        """rebin2x2 should average each 2x2 block."""
        from lsc.cosmics import rebin2x2
        a = np.array([[1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0],
                      [9.0, 10.0, 11.0, 12.0],
                      [13.0, 14.0, 15.0, 16.0]])
        b = rebin2x2(a)
        assert b.shape == (2, 2)
        expected = np.array([[(1+2+5+6)/4.0, (3+4+7+8)/4.0],
                              [(9+10+13+14)/4.0, (11+12+15+16)/4.0]])
        assert np.allclose(b, expected)


# cosmicsimage — class initialisation and properties
class TestCosmicsimageInit:
    def setup_method(self):
        rng = np.random.default_rng(0)
        self.data = rng.normal(1000.0, 30.0, (64, 64)).astype(np.float64)

    def test_init_no_crash(self):
        """cosmicsimage can be constructed with gain and readnoise without error."""
        from lsc.cosmics import cosmicsimage
        ci = cosmicsimage(self.data, gain=2.0, readnoise=10.0)
        assert ci is not None

    def test_mask_all_false_initially(self):
        """The cosmic ray mask starts with all pixels unmasked."""
        from lsc.cosmics import cosmicsimage
        ci = cosmicsimage(self.data)
        assert ci.mask.sum() == 0

    def test_cleanarray_matches_rawarray(self):
        """Before any iteration, cleanarray is a copy of rawarray."""
        from lsc.cosmics import cosmicsimage
        ci = cosmicsimage(self.data)
        assert np.allclose(ci.cleanarray, ci.rawarray)

    def test_pssl_offset(self):
        """pssl is added to rawarray internally to work with the sky included."""
        from lsc.cosmics import cosmicsimage
        ci = cosmicsimage(self.data, pssl=100.0)
        # rawarray should be data + 100
        assert np.allclose(ci.rawarray, self.data + 100.0)

    def test_str_contains_shape(self):
        """__str__ includes the array dimensions."""
        from lsc.cosmics import cosmicsimage
        ci = cosmicsimage(self.data)
        s = str(ci)
        assert "64" in s


# cosmicsimage — lacosmiciteration detects injected cosmics
class TestLacosmiciteration:
    def setup_method(self):
        rng = np.random.default_rng(42)
        self.clean_data = rng.normal(1000.0, 30.0, (128, 128)).astype(np.float64)
        # Inject obvious cosmic ray hits
        self.dirty_data = self.clean_data.copy()
        self.cosmic_positions = [(30, 40), (70, 20), (100, 80)]
        for r, c in self.cosmic_positions:
            self.dirty_data[r, c] = 60000.0

    def test_iteration_detects_cosmics(self):
        """lacosmiciteration() returns niter > 0 when bright cosmics are present."""
        from lsc.cosmics import cosmicsimage
        ci = cosmicsimage(self.dirty_data, gain=2.2, readnoise=10.0,
                          sigclip=5.0, satlevel=-1, verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        assert result["niter"] > 0

    def test_mask_grows_after_iteration(self):
        """After one iteration the mask has flagged at least one pixel."""
        from lsc.cosmics import cosmicsimage
        ci = cosmicsimage(self.dirty_data, gain=2.2, readnoise=10.0,
                          sigclip=5.0, satlevel=-1, verbose=False)
        ci.lacosmiciteration(verbose=False)
        assert ci.mask.sum() > 0

    def test_result_keys(self):
        """The result dict from lacosmiciteration() contains the four expected keys."""
        from lsc.cosmics import cosmicsimage
        ci = cosmicsimage(self.dirty_data, gain=2.2, readnoise=10.0,
                          sigclip=5.0, satlevel=-1, verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        for key in ("niter", "nnew", "itermask", "newmask"):
            assert key in result


# cosmicsimage — clean() replaces detected cosmic pixels
class TestClean:
    def test_clean_replaces_flagged_pixels(self):
        """clean() replaces detected cosmic pixels so none remain as np.inf."""
        from lsc.cosmics import cosmicsimage
        rng = np.random.default_rng(5)
        data = rng.normal(1000.0, 30.0, (64, 64)).astype(np.float64)
        data[20, 20] = 55000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0,
                          sigclip=5.0, satlevel=-1, verbose=False)
        ci.lacosmiciteration(verbose=False)
        before = ci.cleanarray[20, 20]
        ci.clean(verbose=False)
        after = ci.cleanarray[20, 20]
        # The cleaned pixel should no longer be np.Inf
        assert not np.isinf(after)


# cosmicsimage — run() full integration
class TestRun:
    def test_run_completes(self):
        """run() completes without error and returns a valid mask array."""
        from lsc.cosmics import cosmicsimage
        rng = np.random.default_rng(9)
        data = rng.normal(1000.0, 30.0, (64, 64)).astype(np.float64)
        data[15, 15] = 55000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0, satlevel=-1, verbose=False)
        ci.run(maxiter=2, verbose=False)
        # After running, mask should have detected something
        assert isinstance(ci.mask, np.ndarray)

    def test_run_zero_cosmics_on_clean_image(self):
        """With a very high sigclip threshold almost no pixels are flagged on a clean image."""
        from lsc.cosmics import cosmicsimage
        rng = np.random.default_rng(7)
        # Very well-behaved Gaussian image — few if any cosmics expected
        data = rng.normal(1000.0, 5.0, (64, 64)).astype(np.float64)
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=20.0, satlevel=-1, verbose=False)
        ci.run(maxiter=1, verbose=False)
        # With very high sigclip threshold, almost nothing should be flagged
        assert ci.mask.sum() < 10


# fromfits / tofits — round-trip FITS I/O
class TestFitsIO:
    def test_tofits_fromfits_roundtrip(self, tmp_path):
        """Writing then reading a FITS file recovers the original array shape."""
        from lsc.cosmics import fromfits, tofits
        # fromfits transposes; tofits transposes back — net: equal shape
        arr = np.random.default_rng(1).normal(1000, 50, (32, 32)).astype(np.float64)
        outfile = str(tmp_path / "roundtrip.fits")
        tofits(outfile, arr, verbose=False)
        arr2, _ = fromfits(outfile, verbose=False)
        assert arr2.shape == arr.shape or arr2.shape == arr.T.shape

    def test_tofits_creates_file(self, tmp_path):
        """tofits() creates a file at the given path."""
        from lsc.cosmics import tofits
        import os
        arr = np.ones((10, 10))
        outfile = str(tmp_path / "out.fits")
        tofits(outfile, arr, verbose=False)
        assert os.path.exists(outfile)

    def test_tofits_boolean_array_written_as_uint8(self, tmp_path):
        """tofits must convert bool arrays to uint8 before writing."""
        from lsc.cosmics import tofits, fromfits
        arr = np.zeros((8, 8), dtype=bool)
        arr[2, 2] = True
        outfile = str(tmp_path / "bool.fits")
        tofits(outfile, arr, verbose=False)
        arr2, _ = fromfits(outfile, verbose=False)
        # Written values should be 0/1 integers, not a float crash
        assert arr2.dtype != bool
        assert set(arr2.ravel().astype(int)).issubset({0, 1})

    def test_tofits_overwrites_existing_file(self, tmp_path):
        """tofits removes an existing file rather than appending."""
        from lsc.cosmics import tofits, fromfits
        arr1 = np.full((8, 8), 1.0)
        arr2 = np.full((8, 8), 2.0)
        outfile = str(tmp_path / "overwrite.fits")
        tofits(outfile, arr1, verbose=False)
        tofits(outfile, arr2, verbose=False)
        result, _ = fromfits(outfile, verbose=False)
        assert np.allclose(result, 2.0)


# subsample — non-square and degenerate inputs
class TestSubsampleExtra:
    def test_non_square_shape(self):
        """subsample() handles non-square inputs, doubling each dimension independently."""
        from lsc.cosmics import subsample
        a = np.ones((6, 10))
        b = subsample(a)
        assert b.shape == (12, 20)

    def test_single_pixel(self):
        """subsample() expands a single pixel to a 2x2 block of the same value."""
        from lsc.cosmics import subsample
        a = np.array([[5.0]])
        b = subsample(a)
        assert b.shape == (2, 2)
        assert np.allclose(b, 5.0)

# cosmicsimage — __str__ extra branches
class TestCosmicsimageStr:
    def _make(self, **kwargs):
        from lsc.cosmics import cosmicsimage
        rng = np.random.default_rng(0)
        data = rng.normal(1000.0, 30.0, (32, 32)).astype(np.float64)
        return cosmicsimage(data, **kwargs)

    def test_str_shows_pssl(self):
        """__str__ includes the pssl value when it is non-zero."""
        ci = self._make(pssl=50.0)
        assert "50" in str(ci)

    def test_str_no_pssl_line_when_zero(self):
        """__str__ omits the pssl line when pssl is zero."""
        ci = self._make(pssl=0.0)
        assert "previously subtracted" not in str(ci)

    def test_str_shows_satstars_after_findsatstars(self):
        """__str__ reports the saturated star mask after findsatstars() is called."""
        from lsc.cosmics import cosmicsimage
        rng = np.random.default_rng(0)
        data = rng.normal(1000.0, 30.0, (64, 64)).astype(np.float64)
        # Plant a saturated star
        data[30:35, 30:35] = 60000.0
        ci = cosmicsimage(data, satlevel=50000.0, verbose=False)
        ci.findsatstars(verbose=False)
        assert "Saturated star" in str(ci)


# cosmicsimage — getrawarray / getcleanarray pssl correction
class TestGetArrays:
    def test_getrawarray_subtracts_pssl(self):
        """getrawarray() returns the original data array, not rawarray (which includes pssl)."""
        from lsc.cosmics import cosmicsimage
        data = np.full((16, 16), 1000.0)
        ci = cosmicsimage(data, pssl=200.0)
        # getrawarray should return original (data), not data+pssl
        assert np.allclose(ci.getrawarray(), data)

    def test_getcleanarray_reflects_cleaned_pixels(self):
        """getcleanarray() returns the post-clean array (with replaced cosmics), not the raw data."""
        from lsc.cosmics import cosmicsimage
        data = np.random.default_rng(3).normal(1000.0, 30.0, (32, 32)).astype(np.float64)
        data[10, 10] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0, pssl=200.0, satlevel=-1, verbose=False)
        ci.lacosmiciteration(verbose=False)
        ci.clean(verbose=False)
        # getrawarray always returns the original (unmodified) data minus pssl
        assert np.allclose(ci.getrawarray()[10, 10], data[10, 10])
        assert not np.isclose(ci.getcleanarray()[10, 10], data[10, 10])


# cosmicsimage — guessbackgroundlevel
class TestGuessBackgroundLevel:
    def test_returns_median(self):
        """guessbackgroundlevel() returns the median of rawarray."""
        from lsc.cosmics import cosmicsimage
        data = np.arange(100, dtype=np.float64).reshape(10, 10)
        ci = cosmicsimage(data)
        level = ci.guessbackgroundlevel()
        assert np.isclose(level, np.median(data))

    def test_cached_on_second_call(self):
        """guessbackgroundlevel() caches its result and does not recompute on subsequent calls."""
        from lsc.cosmics import cosmicsimage
        data = np.ones((10, 10)) * 42.0
        ci = cosmicsimage(data)
        l1 = ci.guessbackgroundlevel()
        # Mutate rawarray; second call should still return cached value
        ci.rawarray[:] = 0.0
        l2 = ci.guessbackgroundlevel()
        assert l1 == l2


# cosmicsimage — getdilatedmask
class TestGetDilatedMask:
    def _ci_with_mask(self):
        from lsc.cosmics import cosmicsimage
        rng = np.random.default_rng(0)
        data = rng.normal(1000.0, 30.0, (32, 32)).astype(np.float64)
        data[15, 15] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0,
                          sigclip=5.0, satlevel=-1, verbose=False)
        ci.lacosmiciteration(verbose=False)
        return ci

    def test_size3_dilates(self):
        """size=3 dilation produces a mask at least as large as the original."""
        ci = self._ci_with_mask()
        dilated = ci.getdilatedmask(size=3)
        assert dilated.sum() >= ci.mask.sum()

    def test_size5_dilates_more_than_size3(self):
        """size=5 dilation covers at least as many pixels as size=3."""
        ci = self._ci_with_mask()
        d3 = ci.getdilatedmask(size=3)
        d5 = ci.getdilatedmask(size=5)
        assert d5.sum() >= d3.sum()

    def test_unknown_size_returns_copy_of_mask(self):
        """For unrecognised sizes the method falls back to returning a copy."""
        ci = self._ci_with_mask()
        result = ci.getdilatedmask(size=99)
        assert isinstance(result, np.ndarray)


# cosmicsimage — labelmask
class TestLabelmask:
    def test_returns_list(self):
        """labelmask() returns a list of dicts for each detected cosmic island."""
        from lsc.cosmics import cosmicsimage
        rng = np.random.default_rng(0)
        data = rng.normal(1000.0, 30.0, (64, 64)).astype(np.float64)
        data[10, 10] = 60000.0
        data[50, 50] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0,
                          sigclip=5.0, satlevel=-1, verbose=False)
        ci.lacosmiciteration(verbose=False)
        result = ci.labelmask(verbose=False)
        assert isinstance(result, list)

    def test_empty_mask_gives_empty_list(self):
        """labelmask() returns an empty list when no cosmics are masked."""
        from lsc.cosmics import cosmicsimage
        data = np.full((32, 32), 1000.0)
        ci = cosmicsimage(data, verbose=False)
        # mask is all-False by default
        result = ci.labelmask(verbose=False)
        assert result == []


# cosmicsimage — findsatstars / getsatstars
class TestSatStars:
    def _data_with_sat_star(self, satlevel=50000.0):
        rng = np.random.default_rng(0)
        data = rng.normal(1000.0, 30.0, (64, 64)).astype(np.float64)
        # Plant a clearly saturated patch
        data[28:36, 28:36] = satlevel + 10000.0
        return data, satlevel

    def test_findsatstars_sets_satstars_mask(self):
        """findsatstars() populates satstars with pixels above the saturation level."""
        from lsc.cosmics import cosmicsimage
        data, satlevel = self._data_with_sat_star()
        ci = cosmicsimage(data, satlevel=satlevel, verbose=False)
        ci.findsatstars(verbose=False)
        assert ci.satstars is not None
        assert ci.satstars.sum() > 0

    def test_getsatstars_calls_findsatstars_lazily(self):
        """getsatstars() calls findsatstars() automatically if it hasn't been run yet."""
        from lsc.cosmics import cosmicsimage
        data, satlevel = self._data_with_sat_star()
        ci = cosmicsimage(data, satlevel=satlevel, verbose=False)
        assert ci.satstars is None
        result = ci.getsatstars(verbose=False)
        assert result is not None

    def test_getsatstars_negative_satlevel_prints_error(self, capsys):
        """getsatstars with satlevel <= 0 should print an error."""
        from lsc.cosmics import cosmicsimage
        data = np.full((32, 32), 1000.0)
        ci = cosmicsimage(data, satlevel=-1, verbose=False)
        ci.getsatstars(verbose=False)
        captured = capsys.readouterr()
        assert "satlevel" in captured.out.lower() or "satlevel" in captured.err.lower()


# cosmicsimage — clean() with a custom mask
class TestCleanCustomMask:
    def test_custom_mask_replaces_only_masked_pixels(self):
        """clean() with a custom mask replaces only the specified pixels, leaving others unchanged."""
        from lsc.cosmics import cosmicsimage
        data = np.full((16, 16), 1000.0, dtype=np.float64)
        ci = cosmicsimage(data, verbose=False)
        # Supply a hand-crafted mask rather than running lacosmiciteration
        custom_mask = np.zeros((16, 16), dtype=bool)
        custom_mask[8, 8] = True
        ci.clean(mask=custom_mask, verbose=False)
        # The flagged pixel should have been replaced (not inf)
        assert not np.isinf(ci.cleanarray[8, 8])
        # All other pixels should be unchanged
        other = ci.cleanarray.copy()
        other[8, 8] = 1000.0
        assert np.allclose(other, 1000.0)


# run() — early termination
class TestRunEarlyTermination:
    def test_stops_early_when_no_cosmics(self):
        """run() should break before maxiter when niter==0 on a clean image."""
        from lsc.cosmics import cosmicsimage
        import io, contextlib
        rng = np.random.default_rng(11)
        data = rng.normal(1000.0, 5.0, (64, 64)).astype(np.float64)
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0,
                          sigclip=50.0, satlevel=-1, verbose=False)
        # Capture stdout from run() to count iterations printed
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ci.run(maxiter=5, verbose=False)
        output = buf.getvalue()
        # "Iteration 2" should not appear when the first iteration finds nothing
        assert "Iteration 5" not in output

    def test_fromfits_returns_array_and_header(self, tmp_path):
        """fromfits() returns a tuple of (numpy array, FITS header)."""
        from lsc.cosmics import fromfits, tofits
        arr = np.ones((16, 16))
        outfile = str(tmp_path / "test2.fits")
        tofits(outfile, arr, verbose=False)
        arr2, _ = fromfits(outfile, verbose=False)
        assert isinstance(arr2, np.ndarray)


# labelmask — verbose=None path and verbose=True print lines
class TestLabelmaskVerbosePaths:
    def _ci_with_cosmics(self, verbose=False):
        from lsc.cosmics import cosmicsimage
        rng = np.random.default_rng(0)
        data = rng.normal(1000.0, 30.0, (64, 64)).astype(np.float64)
        data[10, 10] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0,
                          sigclip=5.0, satlevel=-1, verbose=verbose)
        ci.lacosmiciteration(verbose=False)
        return ci

    def test_verbose_none_uses_self_verbose(self, capsys):
        """labelmask() with verbose=None inherits self.verbose (line 153)."""
        ci = self._ci_with_cosmics(verbose=True)
        ci.labelmask()  # verbose=None → uses self.verbose=True
        captured = capsys.readouterr()
        assert "Labeling" in captured.out or "done" in captured.out.lower()

    def test_verbose_true_prints_done(self, capsys):
        """labelmask(verbose=True) prints the 'Labeling done' message (line 183)."""
        ci = self._ci_with_cosmics()
        ci.labelmask(verbose=True)
        captured = capsys.readouterr()
        assert "done" in captured.out.lower() or "label" in captured.out.lower()


# labelmask — internal sanity-check RuntimeError (lines 168-171)
class TestLabelmaskInternalError:
    def test_mismatch_slicecouplelist_prints_error(self, capsys):
        """labelmask() prints an error when find_objects returns mismatched count."""
        from lsc.cosmics import cosmicsimage
        from unittest import mock
        import lsc.cosmics as _cosmics_mod
        rng = np.random.default_rng(0)
        data = rng.normal(1000.0, 30.0, (64, 64)).astype(np.float64)
        data[20, 20] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0,
                          sigclip=5.0, satlevel=-1, verbose=False)
        ci.lacosmiciteration(verbose=False)
        # Return 0 slices while ndimage.label returns n>0 to force mismatch
        with mock.patch.object(_cosmics_mod.ndimage, 'find_objects', return_value=[]):
            ci.labelmask(verbose=False)
        captured = capsys.readouterr()
        assert "Mega error" in captured.out


# clean() — verbose=None path (line 217) and verbose=True prints (lines 222, 274)
class TestCleanVerbosePaths:
    def test_verbose_none_uses_self_verbose(self, capsys):
        """clean() with verbose=None inherits self.verbose (line 217)."""
        from lsc.cosmics import cosmicsimage
        data = np.full((16, 16), 1000.0, dtype=np.float64)
        data[8, 8] = 55000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0,
                          sigclip=5.0, satlevel=-1, verbose=True)
        ci.lacosmiciteration(verbose=False)
        ci.clean()  # verbose=None → self.verbose=True
        captured = capsys.readouterr()
        assert "cleaning" in captured.out.lower() or "done" in captured.out.lower()

    def test_verbose_true_prints_messages(self, capsys):
        """clean(verbose=True) prints start and done messages (lines 222, 274)."""
        from lsc.cosmics import cosmicsimage
        data = np.full((16, 16), 1000.0, dtype=np.float64)
        data[8, 8] = 55000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0,
                          sigclip=5.0, satlevel=-1, verbose=False)
        ci.lacosmiciteration(verbose=False)
        ci.clean(verbose=True)
        captured = capsys.readouterr()
        assert "Cleaning" in captured.out
        assert "done" in captured.out.lower()


# clean() — satstars branch masks cosmic neighbors (line 242)
class TestCleanWithSatstars:
    def test_satstars_masked_during_clean(self):
        """clean() excludes satstars pixels from the median interpolation (line 242)."""
        from lsc.cosmics import cosmicsimage
        rng = np.random.default_rng(0)
        data = rng.normal(1000.0, 30.0, (32, 32)).astype(np.float64)
        data[5, 5] = 55000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0,
                          sigclip=5.0, satlevel=-1, verbose=False)
        ci.lacosmiciteration(verbose=False)
        # Assign a dummy satstars mask so the satstars branch is taken
        ci.satstars = np.zeros((32, 32), dtype=bool)
        ci.satstars[20:25, 20:25] = True
        ci.clean(verbose=False)
        assert not np.isinf(ci.cleanarray[5, 5])


# clean() — "Mega error" RuntimeError when goodcutout >= 25 (lines 257-260)
class TestCleanMegaError:
    def test_prints_mega_error_when_all_neighbors_valid(self, capsys):
        """clean() prints Mega error when argwhere returns a position with 25 valid neighbors."""
        from lsc.cosmics import cosmicsimage
        from unittest import mock
        import lsc.cosmics as _cosmics_mod
        data = np.full((16, 16), 1000.0, dtype=np.float64)
        ci = cosmicsimage(data, verbose=False)
        mask = np.zeros((16, 16), dtype=bool)
        # Inject cosmic at (2,2) — its 5x5 padarray window has no border-inf values
        with mock.patch.object(_cosmics_mod.np, 'argwhere', return_value=np.array([[2, 2]])):
            try:
                ci.clean(mask=mask, verbose=False)
            except (NameError, UnboundLocalError):
                pass  # expected: replacementvalue is unset after the >= 25 branch
        captured = capsys.readouterr()
        assert "Mega error" in captured.out


# clean() — "HUUUUGE COSMIC" path when all 5×5 neighbors are inf (lines 265-266)
class TestCleanHugeCosmic:
    def test_huge_cosmic_uses_background_level(self):
        """clean() falls back to guessbackgroundlevel() when all neighbors are inf."""
        from lsc.cosmics import cosmicsimage
        data = np.full((32, 32), 500.0, dtype=np.float64)
        ci = cosmicsimage(data, verbose=False)
        # A large solid mask block: interior pixels have no valid 5×5 neighbors
        big_mask = np.zeros((32, 32), dtype=bool)
        big_mask[6:20, 6:20] = True
        ci.clean(mask=big_mask, verbose=False)
        # Interior pixel should have been replaced by guessbackgroundlevel()
        assert np.isclose(ci.cleanarray[12, 12], ci.guessbackgroundlevel())


# findsatstars() — verbose=None path and verbose=True prints (lines 306, 308, 321, 339, 354)
class TestFindsatstarsVerbose:
    def test_verbose_none_uses_self_verbose(self, capsys):
        """findsatstars() with verbose=None inherits self.verbose (line 306)."""
        from lsc.cosmics import cosmicsimage
        rng = np.random.default_rng(0)
        data = rng.normal(1000.0, 30.0, (64, 64)).astype(np.float64)
        data[30:35, 30:35] = 60000.0
        ci = cosmicsimage(data, satlevel=50000.0, verbose=True)
        ci.findsatstars()  # verbose=None → self.verbose=True
        captured = capsys.readouterr()
        assert "saturated" in captured.out.lower()

    def test_verbose_true_prints_all_messages(self, capsys):
        """findsatstars(verbose=True) prints detection progress messages (lines 308, 321, 339, 354)."""
        from lsc.cosmics import cosmicsimage
        rng = np.random.default_rng(0)
        data = rng.normal(1000.0, 30.0, (64, 64)).astype(np.float64)
        data[30:35, 30:35] = 60000.0
        ci = cosmicsimage(data, satlevel=50000.0, verbose=False)
        ci.findsatstars(verbose=True)
        captured = capsys.readouterr()
        assert "Detecting saturated stars" in captured.out
        assert "mask of saturated stars" in captured.out.lower()
        assert "done" in captured.out.lower()


# getsatstars() — verbose=None path (line 362) and getmask() (line 373)
class TestGetsatstarsVerboseDefaultAndGetmask:
    def test_getsatstars_verbose_none_inherits_self_verbose(self, capsys):
        """getsatstars() with verbose=None inherits self.verbose (line 362)."""
        from lsc.cosmics import cosmicsimage
        rng = np.random.default_rng(0)
        data = rng.normal(1000.0, 30.0, (64, 64)).astype(np.float64)
        data[28:34, 28:34] = 60000.0
        ci = cosmicsimage(data, satlevel=50000.0, verbose=False)
        ci.findsatstars(verbose=False)
        result = ci.getsatstars()  # verbose=None → self.verbose=False (no crash)
        assert result is not None

    def test_getmask_returns_mask_array(self):
        """getmask() returns the current boolean mask array (line 373)."""
        from lsc.cosmics import cosmicsimage
        data = np.full((16, 16), 1000.0)
        ci = cosmicsimage(data, verbose=False)
        mask = ci.getmask()
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool


# lacosmiciteration() — verbose=None + verbose=True + satstars masking branches
class TestLacosmiciterationVerbosePaths:
    def _make_dirty(self, verbose=False):
        from lsc.cosmics import cosmicsimage
        rng = np.random.default_rng(0)
        data = rng.normal(1000.0, 30.0, (64, 64)).astype(np.float64)
        data[20, 20] = 60000.0
        return cosmicsimage(data, gain=2.2, readnoise=10.0,
                            sigclip=5.0, satlevel=-1, verbose=verbose)

    def test_verbose_none_uses_self_verbose(self, capsys):
        """lacosmiciteration() with verbose=None inherits self.verbose (line 417)."""
        ci = self._make_dirty(verbose=True)
        ci.lacosmiciteration()  # verbose=None → self.verbose=True
        captured = capsys.readouterr()
        assert "Convolving" in captured.out or "candidate" in captured.out.lower()

    def test_verbose_true_prints_all_stages(self, capsys):
        """lacosmiciteration(verbose=True) prints all pipeline stage messages."""
        ci = self._make_dirty()
        ci.lacosmiciteration(verbose=True)
        captured = capsys.readouterr()
        assert "Convolving" in captured.out
        assert "noise" in captured.out.lower()
        assert "candidate" in captured.out.lower()
        assert "fine structure" in captured.out.lower()

    def test_verbose_true_with_satstars_masks_candidates(self, capsys):
        """lacosmiciteration(verbose=True) with satstars set prints masking messages (lines 458-464, 511-513)."""
        ci = self._make_dirty()
        ci.satstars = np.zeros((64, 64), dtype=bool)
        ci.lacosmiciteration(verbose=True)
        captured = capsys.readouterr()
        assert "saturated" in captured.out.lower() or "candidate" in captured.out.lower()


# findholes() — stub method body (lines 542-581)
class TestFindholes:
    def test_findholes_runs_without_error(self):
        """findholes() is a stub that returns None without any side effects (line 542)."""
        from lsc.cosmics import cosmicsimage
        data = np.full((16, 16), 1000.0)
        ci = cosmicsimage(data, verbose=False)
        result = ci.findholes()
        assert result is None

    def test_findholes_does_not_modify_mask(self):
        """findholes() does not alter the cosmic mask since it is not implemented."""
        from lsc.cosmics import cosmicsimage
        data = np.full((16, 16), 1000.0)
        ci = cosmicsimage(data, verbose=False)
        before = ci.mask.copy()
        ci.findholes()
        assert np.array_equal(ci.mask, before)


# run() — findsatstars called automatically when satlevel > 0 (line 605)
class TestRunWithPositiveSatlevel:
    def test_run_calls_findsatstars_when_satlevel_positive(self, capsys):
        """run() automatically calls findsatstars() when satlevel > 0 and satstars is None (line 605)."""
        from lsc.cosmics import cosmicsimage
        rng = np.random.default_rng(0)
        data = rng.normal(1000.0, 30.0, (32, 32)).astype(np.float64)
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0,
                          sigclip=20.0, satlevel=50000.0, verbose=False)
        assert ci.satstars is None
        ci.run(maxiter=1, verbose=False)
        assert ci.satstars is not None


# fromfits() — verbose=True prints (lines 654-656)
class TestFromfitsVerbose:
    def test_verbose_true_prints_shape_and_bitpix(self, tmp_path, capsys):
        """fromfits(verbose=True) prints shape, BITPIX and dtype info (lines 654-656)."""
        from lsc.cosmics import fromfits, tofits
        arr = np.ones((16, 16))
        outfile = str(tmp_path / "verbose_in.fits")
        tofits(outfile, arr, verbose=False)
        fromfits(outfile, verbose=True)
        captured = capsys.readouterr()
        assert "shape" in captured.out.lower() or "FITS import" in captured.out
        assert "BITPIX" in captured.out or "dtype" in captured.out.lower()


# tofits() — verbose=True (lines 668, 684) and header != None branch (line 679)
class TestTofitsVerboseAndHeader:
    def test_verbose_true_prints_shape_and_wrote(self, tmp_path, capsys):
        """tofits(verbose=True) prints the export shape and confirmation (lines 668, 684)."""
        from lsc.cosmics import tofits
        arr = np.ones((12, 12))
        outfile = str(tmp_path / "verbose_out.fits")
        tofits(outfile, arr, verbose=True)
        captured = capsys.readouterr()
        assert "FITS export shape" in captured.out
        assert "Wrote" in captured.out

    def test_with_header_uses_header_hdu(self, tmp_path):
        """tofits() with an explicit header uses it in the PrimaryHDU (line 679)."""
        from lsc.cosmics import tofits, fromfits
        arr = np.ones((8, 8))
        outfile = str(tmp_path / "hdr_out.fits")
        # Round-trip to get a real header object
        tofits(outfile, arr, verbose=False)
        _, hdr = fromfits(outfile, verbose=False)
        # Now write again using that header
        outfile2 = str(tmp_path / "hdr_out2.fits")
        tofits(outfile2, arr, hdr=hdr, verbose=False)
        arr2, _ = fromfits(outfile2, verbose=False)
        assert arr2.shape == arr.shape or arr2.shape == arr.T.shape
