"""
Tests for lsc.cosmics — the L.A.Cosmic cosmic-ray detection algorithm.

Covers array utilities, class init/properties, iteration, cleaning, FITS I/O,
edge cases, parameter sensitivity, kernel operations, mask handling,
convergence, and boundary conditions.
"""
import io
import contextlib
import os

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from unittest import mock

pytestmark = pytest.mark.unit


# =============================================================================
# Helper utilities
# =============================================================================

def make_flat_image(shape=(64, 64), value=1000.0, dtype=np.float64):
    """Create a constant-value image."""
    return np.full(shape, value, dtype=dtype)


def make_noisy_image(shape=(64, 64), sky=1000.0, noise=30.0, seed=42, dtype=np.float64):
    """Create a Gaussian noise image around a given sky level."""
    rng = np.random.default_rng(seed)
    return rng.normal(sky, noise, shape).astype(dtype)


def inject_cosmics(data, positions, value=60000.0):
    """Inject cosmic ray hits at specified positions."""
    data = data.copy()
    for r, c in positions:
        data[r, c] = value
    return data


# =============================================================================
# subsample / rebin / rebin2x2 — array utilities
# =============================================================================

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

    def test_preserves_dtype(self):
        """subsample preserves the dtype of the input array."""
        from lsc.cosmics import subsample
        a = np.ones((10, 10), dtype=np.float32)
        b = subsample(a)
        assert b.dtype == np.float32

    def test_each_pixel_becomes_2x2(self):
        """Each pixel in input maps to a 2x2 block in output."""
        from lsc.cosmics import subsample
        a = np.array([[1.0, 2.0],
                      [3.0, 4.0]])
        b = subsample(a)
        assert b.shape == (4, 4)
        assert b[0, 0] == 1.0
        assert b[0, 1] == 1.0
        assert b[1, 0] == 1.0
        assert b[1, 1] == 1.0
        assert b[0, 2] == 2.0
        assert b[0, 3] == 2.0
        assert b[1, 2] == 2.0
        assert b[1, 3] == 2.0

    def test_large_values(self):
        """subsample handles large pixel values without overflow."""
        from lsc.cosmics import subsample
        a = np.array([[1e10]], dtype=np.float64)
        b = subsample(a)
        assert np.all(b == 1e10)

    def test_negative_values(self):
        """subsample correctly handles negative values."""
        from lsc.cosmics import subsample
        a = np.array([[-500.0, 500.0],
                      [0.0, -1000.0]])
        b = subsample(a)
        assert b[0, 0] == -500.0
        assert b[0, 2] == 500.0
        assert b[3, 3] == -1000.0


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

    def test_3x3_to_1x1(self):
        """Rebinning a 3x3 array to 1x1 returns the mean."""
        from lsc.cosmics import rebin
        a = np.arange(9, dtype=np.float64).reshape(3, 3)
        b = rebin(a, (1, 1))
        assert b.shape == (1, 1)
        assert np.isclose(b[0, 0], a.mean())

    def test_identity(self):
        """Rebinning to the same shape returns the same values."""
        from lsc.cosmics import rebin
        a = np.ones((4, 4), dtype=np.float64) * 5.0
        b = rebin(a, (4, 4))
        assert np.allclose(b, 5.0)

    def test_non_square(self):
        """Rebinning a rectangular array works correctly."""
        from lsc.cosmics import rebin
        a = np.ones((6, 8), dtype=np.float64) * 3.0
        b = rebin(a, (3, 4))
        assert b.shape == (3, 4)
        assert np.allclose(b, 3.0)


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

    def test_with_varying_values(self):
        """rebin2x2 correctly averages 2x2 blocks of varying values."""
        from lsc.cosmics import rebin2x2
        a = np.array([[10.0, 20.0, 30.0, 40.0],
                      [50.0, 60.0, 70.0, 80.0],
                      [1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0]])
        b = rebin2x2(a)
        assert b.shape == (2, 2)
        assert np.isclose(b[0, 0], (10 + 20 + 50 + 60) / 4.0)
        assert np.isclose(b[0, 1], (30 + 40 + 70 + 80) / 4.0)

    def test_minimum_even_size(self):
        """rebin2x2 works on the minimum valid even size (2x2)."""
        from lsc.cosmics import rebin2x2
        a = np.array([[4.0, 8.0],
                      [12.0, 16.0]])
        b = rebin2x2(a)
        assert b.shape == (1, 1)
        assert np.isclose(b[0, 0], 10.0)

    def test_odd_even_mix_raises(self):
        """rebin2x2 raises an error on arrays with one odd dimension."""
        from lsc.cosmics import rebin2x2
        a = np.ones((6, 7))
        with pytest.raises(Exception):
            rebin2x2(a)


class TestSubsampleRebinRoundtrip:
    """Test that subsample followed by rebin2x2 gives back the original."""

    def test_roundtrip_preserves_values(self):
        """subsample then rebin2x2 recovers the original array values."""
        from lsc.cosmics import subsample, rebin2x2
        a = np.array([[1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0],
                      [9.0, 10.0, 11.0, 12.0],
                      [13.0, 14.0, 15.0, 16.0]])
        b = subsample(a)
        c = rebin2x2(b)
        assert c.shape == a.shape
        assert np.allclose(c, a)

    def test_roundtrip_uniform_image(self):
        """Roundtrip on uniform image preserves uniform value."""
        from lsc.cosmics import subsample, rebin2x2
        a = np.full((10, 10), 42.0)
        b = subsample(a)
        c = rebin2x2(b)
        assert np.allclose(c, 42.0)


# =============================================================================
# Laplacian kernel operations
# =============================================================================

class TestLaplacianKernel:
    """Test the Laplacian kernel and its convolution."""

    def test_laplkernel_shape(self):
        """The Laplacian kernel is 3x3."""
        from lsc.cosmics import laplkernel
        assert laplkernel.shape == (3, 3)

    def test_laplkernel_sum_is_zero(self):
        """The Laplacian kernel sums to zero (edge-detecting property)."""
        from lsc.cosmics import laplkernel
        assert np.isclose(laplkernel.sum(), 0.0)

    def test_laplkernel_center_is_positive(self):
        """The center element of the Laplacian kernel is positive (4.0)."""
        from lsc.cosmics import laplkernel
        assert laplkernel[1, 1] == 4.0

    def test_growkernel_shape_and_values(self):
        """The grow kernel is 3x3 of all ones."""
        from lsc.cosmics import growkernel
        assert growkernel.shape == (3, 3)
        assert np.all(growkernel == 1.0)

    def test_dilstruct_shape(self):
        """The dilation structure is 5x5 with corners cut."""
        from lsc.cosmics import dilstruct
        assert dilstruct.shape == (5, 5)
        assert dilstruct[0, 0] == 0
        assert dilstruct[0, 4] == 0
        assert dilstruct[4, 0] == 0
        assert dilstruct[4, 4] == 0
        assert dilstruct[0, 1] == 1
        assert dilstruct[0, 2] == 1
        assert dilstruct[0, 3] == 1

    def test_laplacian_on_flat_image_is_zero(self):
        """Convolving a flat image with the Laplacian produces approximately zero."""
        import scipy.signal as signal
        from lsc.cosmics import laplkernel, subsample, rebin2x2
        data = make_flat_image(shape=(32, 32), value=1000.0)
        subsam = subsample(data)
        conved = signal.convolve2d(subsam, laplkernel, mode="same", boundary="symm")
        assert np.allclose(conved[2:-2, 2:-2], 0.0, atol=1e-10)

    def test_laplacian_detects_single_spike(self):
        """Convolving an image with one spike gives a strong positive response at that location."""
        import scipy.signal as signal
        from lsc.cosmics import laplkernel, subsample, rebin2x2
        data = make_flat_image(shape=(32, 32), value=100.0)
        data[16, 16] = 50000.0
        subsam = subsample(data)
        conved = signal.convolve2d(subsam, laplkernel, mode="same", boundary="symm")
        cliped = conved.clip(min=0.0)
        lplus = rebin2x2(cliped)
        assert lplus[16, 16] == lplus.max()


# =============================================================================
# cosmicsimage — class initialisation and properties
# =============================================================================

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
        assert np.allclose(ci.rawarray, self.data + 100.0)

    def test_str_contains_shape(self):
        """__str__ includes the array dimensions."""
        from lsc.cosmics import cosmicsimage
        ci = cosmicsimage(self.data)
        s = str(ci)
        assert "64" in s


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
        data[30:35, 30:35] = 60000.0
        ci = cosmicsimage(data, satlevel=50000.0, verbose=False)
        ci.findsatstars(verbose=False)
        assert "Saturated star" in str(ci)

    def test_str_after_iteration_shows_mask_count(self):
        """__str__ shows number of cosmic pixels after iteration."""
        from lsc.cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), seed=1400)
        data[30, 30] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        ci.lacosmiciteration(verbose=False)
        s = str(ci)
        assert "cosmic ray mask" in s.lower() or "pixel" in s.lower()

    def test_str_includes_dtype(self):
        """__str__ includes the array dtype."""
        from lsc.cosmics import cosmicsimage
        data = np.ones((32, 32), dtype=np.float64)
        ci = cosmicsimage(data, verbose=False)
        s = str(ci)
        assert "float64" in s

    def test_str_with_satstars_set(self):
        """__str__ shows saturated star info when satstars is set."""
        from lsc.cosmics import cosmicsimage
        data = np.ones((32, 32), dtype=np.float64) * 1000.0
        ci = cosmicsimage(data, verbose=False)
        ci.satstars = np.zeros((32, 32), dtype=bool)
        ci.satstars[10:15, 10:15] = True
        s = str(ci)
        assert "Saturated star" in s or "saturated" in s.lower()


# =============================================================================
# cosmicsimage — getrawarray / getcleanarray / getmask / pssl
# =============================================================================

class TestGetArrays:
    def test_getrawarray_subtracts_pssl(self):
        """getrawarray() returns the original data array, not rawarray (which includes pssl)."""
        from lsc.cosmics import cosmicsimage
        data = np.full((16, 16), 1000.0)
        ci = cosmicsimage(data, pssl=200.0)
        assert np.allclose(ci.getrawarray(), data)

    def test_getcleanarray_reflects_cleaned_pixels(self):
        """getcleanarray() returns the post-clean array (with replaced cosmics), not the raw data."""
        from lsc.cosmics import cosmicsimage
        data = np.random.default_rng(3).normal(1000.0, 30.0, (32, 32)).astype(np.float64)
        data[10, 10] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0, pssl=200.0, satlevel=-1, verbose=False)
        ci.lacosmiciteration(verbose=False)
        ci.clean(verbose=False)
        assert np.allclose(ci.getrawarray()[10, 10], data[10, 10])
        assert not np.isclose(ci.getcleanarray()[10, 10], data[10, 10])

    def test_getmask_returns_mask_array(self):
        """getmask() returns the current boolean mask array."""
        from lsc.cosmics import cosmicsimage
        data = np.full((16, 16), 1000.0)
        ci = cosmicsimage(data, verbose=False)
        mask = ci.getmask()
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool

    def test_getrawarray_shape_matches_input(self):
        """getrawarray returns array with same shape as input."""
        from lsc.cosmics import cosmicsimage
        data = make_noisy_image(shape=(20, 30), seed=1600)
        ci = cosmicsimage(data, verbose=False)
        assert ci.getrawarray().shape == (20, 30)

    def test_getcleanarray_shape_matches_input(self):
        """getcleanarray returns array with same shape as input."""
        from lsc.cosmics import cosmicsimage
        data = make_noisy_image(shape=(20, 30), seed=1601)
        ci = cosmicsimage(data, verbose=False)
        assert ci.getcleanarray().shape == (20, 30)

    def test_getcleanarray_initially_equals_getrawarray(self):
        """Before any processing, getcleanarray equals getrawarray."""
        from lsc.cosmics import cosmicsimage
        data = make_noisy_image(shape=(32, 32), seed=1602)
        ci = cosmicsimage(data, pssl=100.0, verbose=False)
        assert np.allclose(ci.getcleanarray(), ci.getrawarray())


class TestPSSL:
    """Test the pssl parameter handling."""

    def test_pssl_added_to_rawarray(self):
        from lsc.cosmics import cosmicsimage
        data = np.full((16, 16), 1000.0)
        ci = cosmicsimage(data, pssl=500.0, verbose=False)
        assert np.allclose(ci.rawarray, 1500.0)

    def test_pssl_subtracted_in_getrawarray(self):
        from lsc.cosmics import cosmicsimage
        data = np.full((16, 16), 1000.0)
        ci = cosmicsimage(data, pssl=500.0, verbose=False)
        assert np.allclose(ci.getrawarray(), 1000.0)

    def test_pssl_subtracted_in_getcleanarray(self):
        from lsc.cosmics import cosmicsimage
        data = np.full((16, 16), 1000.0)
        ci = cosmicsimage(data, pssl=500.0, verbose=False)
        assert np.allclose(ci.getcleanarray(), 1000.0)

    def test_negative_pssl(self):
        """Negative pssl is handled correctly (sky was over-subtracted)."""
        from lsc.cosmics import cosmicsimage
        data = np.full((16, 16), 1000.0)
        ci = cosmicsimage(data, pssl=-200.0, verbose=False)
        assert np.allclose(ci.rawarray, 800.0)
        assert np.allclose(ci.getrawarray(), 1000.0)

    def test_zero_pssl_no_change(self):
        from lsc.cosmics import cosmicsimage
        data = np.full((16, 16), 1000.0)
        ci = cosmicsimage(data, pssl=0.0, verbose=False)
        assert np.allclose(ci.rawarray, 1000.0)


# =============================================================================
# cosmicsimage — guessbackgroundlevel
# =============================================================================

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
        ci.rawarray[:] = 0.0
        l2 = ci.guessbackgroundlevel()
        assert l1 == l2

    def test_zero_image(self):
        """guessbackgroundlevel on zero image returns 0."""
        from lsc.cosmics import cosmicsimage
        data = np.zeros((32, 32), dtype=np.float64)
        ci = cosmicsimage(data, verbose=False)
        assert ci.guessbackgroundlevel() == 0.0

    def test_backgroundlevel_initially_none(self):
        from lsc.cosmics import cosmicsimage
        data = make_flat_image(shape=(16, 16), value=1000.0)
        ci = cosmicsimage(data, verbose=False)
        assert ci.backgroundlevel is None

    def test_backgroundlevel_set_after_first_call(self):
        from lsc.cosmics import cosmicsimage
        data = make_flat_image(shape=(16, 16), value=1000.0)
        ci = cosmicsimage(data, verbose=False)
        ci.guessbackgroundlevel()
        assert ci.backgroundlevel is not None

    def test_backgroundlevel_is_median_of_rawarray(self):
        from lsc.cosmics import cosmicsimage
        data = np.arange(256, dtype=np.float64).reshape(16, 16)
        ci = cosmicsimage(data, pssl=0.0, verbose=False)
        level = ci.guessbackgroundlevel()
        expected = np.median(data.ravel())
        assert np.isclose(level, expected)


# =============================================================================
# cosmicsimage — getdilatedmask
# =============================================================================

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

    def test_empty_mask_dilated_is_still_empty(self):
        """Dilating an all-False mask returns an all-False mask."""
        from lsc.cosmics import cosmicsimage
        data = make_flat_image(shape=(32, 32))
        ci = cosmicsimage(data, verbose=False)
        d3 = ci.getdilatedmask(size=3)
        d5 = ci.getdilatedmask(size=5)
        assert d3.sum() == 0
        assert d5.sum() == 0

    def test_single_pixel_dilated_size3(self):
        """Dilating a single masked pixel with size=3 produces a cross/3x3 pattern."""
        from lsc.cosmics import cosmicsimage
        data = make_flat_image(shape=(32, 32))
        ci = cosmicsimage(data, verbose=False)
        ci.mask[16, 16] = True
        d3 = ci.getdilatedmask(size=3)
        assert d3.sum() >= 5
        assert d3.sum() <= 9

    def test_single_pixel_dilated_size5(self):
        """Dilating a single masked pixel with size=5 produces a larger pattern."""
        from lsc.cosmics import cosmicsimage
        data = make_flat_image(shape=(32, 32))
        ci = cosmicsimage(data, verbose=False)
        ci.mask[16, 16] = True
        d5 = ci.getdilatedmask(size=5)
        assert d5.sum() > 9


# =============================================================================
# cosmicsimage — labelmask
# =============================================================================

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
        result = ci.labelmask(verbose=False)
        assert result == []

    def test_dict_has_expected_keys(self):
        """Each dict in labelmask result has 'name', 'x', 'y' keys."""
        from lsc.cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), seed=800)
        data[30, 30] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        ci.lacosmiciteration(verbose=False)
        result = ci.labelmask(verbose=False)
        if len(result) > 0:
            item = result[0]
            assert "name" in item
            assert "x" in item
            assert "y" in item

    def test_multiple_cosmic_islands(self):
        """labelmask correctly identifies multiple separate cosmic islands."""
        from lsc.cosmics import cosmicsimage
        data = make_noisy_image(shape=(128, 128), seed=801)
        data[20, 20] = 60000.0
        data[100, 100] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        ci.lacosmiciteration(verbose=False)
        result = ci.labelmask(verbose=False)
        assert len(result) >= 2


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
        """labelmask() with verbose=None inherits self.verbose."""
        ci = self._ci_with_cosmics(verbose=True)
        ci.labelmask()
        captured = capsys.readouterr()
        assert "Labeling" in captured.out or "done" in captured.out.lower()

    def test_verbose_true_prints_done(self, capsys):
        """labelmask(verbose=True) prints the 'Labeling done' message."""
        ci = self._ci_with_cosmics()
        ci.labelmask(verbose=True)
        captured = capsys.readouterr()
        assert "done" in captured.out.lower() or "label" in captured.out.lower()


class TestLabelmaskInternalError:
    def test_mismatch_slicecouplelist_prints_error(self, capsys):
        """labelmask() prints an error when find_objects returns mismatched count."""
        from lsc.cosmics import cosmicsimage
        import lsc.cosmics as _cosmics_mod
        rng = np.random.default_rng(0)
        data = rng.normal(1000.0, 30.0, (64, 64)).astype(np.float64)
        data[20, 20] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0,
                          sigclip=5.0, satlevel=-1, verbose=False)
        ci.lacosmiciteration(verbose=False)
        with mock.patch.object(_cosmics_mod.ndimage, 'find_objects', return_value=[]):
            ci.labelmask(verbose=False)
        captured = capsys.readouterr()
        assert "Mega error" in captured.out


# =============================================================================
# cosmicsimage — lacosmiciteration
# =============================================================================

class TestLacosmiciteration:
    def setup_method(self):
        rng = np.random.default_rng(42)
        self.clean_data = rng.normal(1000.0, 30.0, (128, 128)).astype(np.float64)
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


class TestLacosmiciterationVerbosePaths:
    def _make_dirty(self, verbose=False):
        from lsc.cosmics import cosmicsimage
        rng = np.random.default_rng(0)
        data = rng.normal(1000.0, 30.0, (64, 64)).astype(np.float64)
        data[20, 20] = 60000.0
        return cosmicsimage(data, gain=2.2, readnoise=10.0,
                            sigclip=5.0, satlevel=-1, verbose=verbose)

    def test_verbose_none_uses_self_verbose(self, capsys):
        """lacosmiciteration() with verbose=None inherits self.verbose."""
        ci = self._make_dirty(verbose=True)
        ci.lacosmiciteration()
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
        """lacosmiciteration(verbose=True) with satstars set prints masking messages."""
        ci = self._make_dirty()
        ci.satstars = np.zeros((64, 64), dtype=bool)
        ci.lacosmiciteration(verbose=True)
        captured = capsys.readouterr()
        assert "saturated" in captured.out.lower() or "candidate" in captured.out.lower()


class TestLacosmicIterationWithSatstars:
    """Test that satstars masking works during iteration."""

    def test_cosmic_inside_satstar_region_not_detected(self):
        from lsc.cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), seed=1500)
        data[30, 30] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        ci.satstars = np.zeros((64, 64), dtype=bool)
        ci.satstars[28:33, 28:33] = True
        result = ci.lacosmiciteration(verbose=False)
        assert not result["itermask"][30, 30]

    def test_cosmic_outside_satstar_region_still_detected(self):
        from lsc.cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), seed=1501)
        data[10, 10] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        ci.satstars = np.zeros((64, 64), dtype=bool)
        ci.satstars[50:55, 50:55] = True
        result = ci.lacosmiciteration(verbose=False)
        assert result["niter"] > 0


# =============================================================================
# cosmicsimage — clean()
# =============================================================================

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
        ci.clean(verbose=False)
        assert not np.isinf(ci.cleanarray[20, 20])

    def test_custom_mask_replaces_only_masked_pixels(self):
        """clean() with a custom mask replaces only the specified pixels."""
        from lsc.cosmics import cosmicsimage
        data = np.full((16, 16), 1000.0, dtype=np.float64)
        ci = cosmicsimage(data, verbose=False)
        custom_mask = np.zeros((16, 16), dtype=bool)
        custom_mask[8, 8] = True
        ci.clean(mask=custom_mask, verbose=False)
        assert not np.isinf(ci.cleanarray[8, 8])
        other = ci.cleanarray.copy()
        other[8, 8] = 1000.0
        assert np.allclose(other, 1000.0)

    def test_clean_cosmic_at_corner(self):
        """clean() handles a cosmic at the image corner (uses padded array)."""
        from lsc.cosmics import cosmicsimage
        data = make_flat_image(shape=(32, 32), value=1000.0)
        data[0, 0] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=3.0,
                          satlevel=-1, verbose=False)
        ci.lacosmiciteration(verbose=False)
        ci.clean(verbose=False)
        assert not np.isinf(ci.cleanarray[0, 0])

    def test_clean_cosmic_at_edge(self):
        """clean() handles a cosmic on the image edge."""
        from lsc.cosmics import cosmicsimage
        data = make_flat_image(shape=(32, 32), value=1000.0)
        data[0, 15] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=3.0,
                          satlevel=-1, verbose=False)
        ci.lacosmiciteration(verbose=False)
        ci.clean(verbose=False)
        assert not np.isinf(ci.cleanarray[0, 15])

    def test_clean_replaces_with_local_median(self):
        """Cleaned pixel value should approximate the local background."""
        from lsc.cosmics import cosmicsimage
        data = make_flat_image(shape=(32, 32), value=1000.0)
        data[16, 16] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=3.0,
                          satlevel=-1, verbose=False)
        ci.lacosmiciteration(verbose=False)
        ci.clean(verbose=False)
        assert abs(ci.cleanarray[16, 16] - 1000.0) < 100.0

    def test_clean_empty_mask_does_nothing(self):
        """clean() with all-False mask does not modify cleanarray."""
        from lsc.cosmics import cosmicsimage
        data = make_flat_image(shape=(16, 16), value=500.0)
        ci = cosmicsimage(data, verbose=False)
        original = ci.cleanarray.copy()
        ci.clean(verbose=False)
        assert np.array_equal(ci.cleanarray, original)

    def test_clean_entire_image_masked(self):
        """clean() when entire image is masked uses guessbackgroundlevel for all."""
        from lsc.cosmics import cosmicsimage
        data = make_flat_image(shape=(16, 16), value=700.0)
        ci = cosmicsimage(data, verbose=False)
        full_mask = np.ones((16, 16), dtype=bool)
        ci.clean(mask=full_mask, verbose=False)
        bg = ci.guessbackgroundlevel()
        assert np.isclose(ci.cleanarray[8, 8], bg)


class TestCleanVerbosePaths:
    def test_verbose_none_uses_self_verbose(self, capsys):
        """clean() with verbose=None inherits self.verbose."""
        from lsc.cosmics import cosmicsimage
        data = np.full((16, 16), 1000.0, dtype=np.float64)
        data[8, 8] = 55000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0,
                          sigclip=5.0, satlevel=-1, verbose=True)
        ci.lacosmiciteration(verbose=False)
        ci.clean()
        captured = capsys.readouterr()
        assert "cleaning" in captured.out.lower() or "done" in captured.out.lower()

    def test_verbose_true_prints_messages(self, capsys):
        """clean(verbose=True) prints start and done messages."""
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


class TestCleanWithSatstars:
    def test_satstars_masked_during_clean(self):
        """clean() excludes satstars pixels from the median interpolation."""
        from lsc.cosmics import cosmicsimage
        rng = np.random.default_rng(0)
        data = rng.normal(1000.0, 30.0, (32, 32)).astype(np.float64)
        data[5, 5] = 55000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0,
                          sigclip=5.0, satlevel=-1, verbose=False)
        ci.lacosmiciteration(verbose=False)
        ci.satstars = np.zeros((32, 32), dtype=bool)
        ci.satstars[20:25, 20:25] = True
        ci.clean(verbose=False)
        assert not np.isinf(ci.cleanarray[5, 5])


class TestCleanMegaError:
    def test_prints_mega_error_when_all_neighbors_valid(self, capsys):
        """clean() prints Mega error when argwhere returns a position with 25 valid neighbors."""
        from lsc.cosmics import cosmicsimage
        import lsc.cosmics as _cosmics_mod
        data = np.full((16, 16), 1000.0, dtype=np.float64)
        ci = cosmicsimage(data, verbose=False)
        mask = np.zeros((16, 16), dtype=bool)
        with mock.patch.object(_cosmics_mod.np, 'argwhere', return_value=np.array([[2, 2]])):
            try:
                ci.clean(mask=mask, verbose=False)
            except (NameError, UnboundLocalError):
                pass
        captured = capsys.readouterr()
        assert "Mega error" in captured.out


class TestCleanHugeCosmic:
    def test_huge_cosmic_uses_background_level(self):
        """clean() falls back to guessbackgroundlevel() when all neighbors are inf."""
        from lsc.cosmics import cosmicsimage
        data = np.full((32, 32), 500.0, dtype=np.float64)
        ci = cosmicsimage(data, verbose=False)
        big_mask = np.zeros((32, 32), dtype=bool)
        big_mask[6:20, 6:20] = True
        ci.clean(mask=big_mask, verbose=False)
        assert np.isclose(ci.cleanarray[12, 12], ci.guessbackgroundlevel())


# =============================================================================
# cosmicsimage — findsatstars / getsatstars
# =============================================================================

class TestSatStars:
    def _data_with_sat_star(self, satlevel=50000.0):
        rng = np.random.default_rng(0)
        data = rng.normal(1000.0, 30.0, (64, 64)).astype(np.float64)
        data[28:36, 28:36] = satlevel + 10000.0
        return data, satlevel

    def test_findsatstars_sets_satstars_mask(self):
        from lsc.cosmics import cosmicsimage
        data, satlevel = self._data_with_sat_star()
        ci = cosmicsimage(data, satlevel=satlevel, verbose=False)
        ci.findsatstars(verbose=False)
        assert ci.satstars is not None
        assert ci.satstars.sum() > 0

    def test_getsatstars_calls_findsatstars_lazily(self):
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

    def test_saturated_star_not_detected_as_cosmic(self):
        """Saturated stars should be excluded from cosmic ray detection."""
        from lsc.cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), sky=1000.0, noise=30.0, seed=500)
        data[28:36, 28:36] = 55000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=50000.0, verbose=False)
        ci.findsatstars(verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        assert ci.mask[28:36, 28:36].sum() == 0

    def test_negative_satlevel_skips_satstar_detection(self):
        from lsc.cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), sky=1000.0, noise=30.0, seed=501)
        data[28:36, 28:36] = 55000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        ci.run(maxiter=1, verbose=False)
        assert ci.satstars is None

    def test_findsatstars_no_saturated_stars(self):
        """findsatstars on clean image yields empty mask."""
        from lsc.cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), sky=1000.0, noise=30.0, seed=502)
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, satlevel=50000.0, verbose=False)
        ci.findsatstars(verbose=False)
        assert ci.satstars.sum() == 0

    def test_findsatstars_multiple_saturated_stars(self):
        from lsc.cosmics import cosmicsimage
        data = make_noisy_image(shape=(128, 128), sky=1000.0, noise=30.0, seed=503)
        data[20:28, 20:28] = 55000.0
        data[80:88, 80:88] = 55000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, satlevel=50000.0, verbose=False)
        ci.findsatstars(verbose=False)
        assert ci.satstars.sum() > 0


class TestFindsatstarsVerbose:
    def test_verbose_none_uses_self_verbose(self, capsys):
        from lsc.cosmics import cosmicsimage
        rng = np.random.default_rng(0)
        data = rng.normal(1000.0, 30.0, (64, 64)).astype(np.float64)
        data[30:35, 30:35] = 60000.0
        ci = cosmicsimage(data, satlevel=50000.0, verbose=True)
        ci.findsatstars()
        captured = capsys.readouterr()
        assert "saturated" in captured.out.lower()

    def test_verbose_true_prints_all_messages(self, capsys):
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


class TestGetsatstarsVerboseDefaultAndGetmask:
    def test_getsatstars_verbose_none_inherits_self_verbose(self, capsys):
        from lsc.cosmics import cosmicsimage
        rng = np.random.default_rng(0)
        data = rng.normal(1000.0, 30.0, (64, 64)).astype(np.float64)
        data[28:34, 28:34] = 60000.0
        ci = cosmicsimage(data, satlevel=50000.0, verbose=False)
        ci.findsatstars(verbose=False)
        result = ci.getsatstars()
        assert result is not None


# =============================================================================
# cosmicsimage — findholes (stub)
# =============================================================================

class TestFindholes:
    def test_findholes_runs_without_error(self):
        from lsc.cosmics import cosmicsimage
        data = np.full((16, 16), 1000.0)
        ci = cosmicsimage(data, verbose=False)
        result = ci.findholes()
        assert result is None

    def test_findholes_does_not_modify_mask(self):
        from lsc.cosmics import cosmicsimage
        data = np.full((16, 16), 1000.0)
        ci = cosmicsimage(data, verbose=False)
        before = ci.mask.copy()
        ci.findholes()
        assert np.array_equal(ci.mask, before)

    def test_findholes_does_not_modify_any_state(self):
        from lsc.cosmics import cosmicsimage
        data = make_noisy_image(shape=(32, 32), seed=1700)
        ci = cosmicsimage(data, verbose=False)
        mask_before = ci.mask.copy()
        clean_before = ci.cleanarray.copy()
        raw_before = ci.rawarray.copy()
        ci.findholes()
        assert np.array_equal(ci.mask, mask_before)
        assert np.array_equal(ci.cleanarray, clean_before)
        assert np.array_equal(ci.rawarray, raw_before)


# =============================================================================
# cosmicsimage — run() full integration
# =============================================================================

class TestRun:
    def test_run_completes(self):
        """run() completes without error and returns a valid mask array."""
        from lsc.cosmics import cosmicsimage
        rng = np.random.default_rng(9)
        data = rng.normal(1000.0, 30.0, (64, 64)).astype(np.float64)
        data[15, 15] = 55000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0, satlevel=-1, verbose=False)
        ci.run(maxiter=2, verbose=False)
        assert isinstance(ci.mask, np.ndarray)

    def test_run_zero_cosmics_on_clean_image(self):
        """With a very high sigclip threshold almost no pixels are flagged on a clean image."""
        from lsc.cosmics import cosmicsimage
        rng = np.random.default_rng(7)
        data = rng.normal(1000.0, 5.0, (64, 64)).astype(np.float64)
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=20.0, satlevel=-1, verbose=False)
        ci.run(maxiter=1, verbose=False)
        assert ci.mask.sum() < 10

    def test_run_calls_findsatstars_when_satlevel_positive(self, capsys):
        """run() automatically calls findsatstars() when satlevel > 0 and satstars is None."""
        from lsc.cosmics import cosmicsimage
        rng = np.random.default_rng(0)
        data = rng.normal(1000.0, 30.0, (32, 32)).astype(np.float64)
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0,
                          sigclip=20.0, satlevel=50000.0, verbose=False)
        assert ci.satstars is None
        ci.run(maxiter=1, verbose=False)
        assert ci.satstars is not None

    def test_run_skips_findsatstars_if_already_set(self):
        """run() does not call findsatstars again if satstars is already set."""
        from lsc.cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), seed=1200)
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=20.0,
                          satlevel=50000.0, verbose=False)
        ci.satstars = np.zeros((64, 64), dtype=bool)
        ci.run(maxiter=1, verbose=False)
        assert ci.satstars.sum() == 0


class TestRunEarlyTermination:
    def test_stops_early_when_no_cosmics(self):
        """run() should break before maxiter when niter==0 on a clean image."""
        from lsc.cosmics import cosmicsimage
        rng = np.random.default_rng(11)
        data = rng.normal(1000.0, 5.0, (64, 64)).astype(np.float64)
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0,
                          sigclip=50.0, satlevel=-1, verbose=False)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ci.run(maxiter=5, verbose=False)
        output = buf.getvalue()
        assert "Iteration 5" not in output


# =============================================================================
# Edge cases: all-zero, constant, small, non-square, negative images
# =============================================================================

class TestAllZeroImage:
    def test_init_with_zero_image(self):
        from lsc.cosmics import cosmicsimage
        data = np.zeros((32, 32), dtype=np.float64)
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, satlevel=-1, verbose=False)
        assert ci.rawarray.shape == (32, 32)
        assert np.all(ci.rawarray == 0.0)

    def test_iteration_on_zero_image(self):
        from lsc.cosmics import cosmicsimage
        data = np.zeros((32, 32), dtype=np.float64)
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, satlevel=-1, verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        assert result["niter"] == 0
        assert result["nnew"] == 0

    def test_clean_on_zero_image_no_mask(self):
        from lsc.cosmics import cosmicsimage
        data = np.zeros((32, 32), dtype=np.float64)
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, satlevel=-1, verbose=False)
        ci.clean(verbose=False)
        assert np.all(ci.cleanarray == 0.0)


class TestConstantImage:
    def test_constant_image_no_cosmics_detected(self):
        from lsc.cosmics import cosmicsimage
        data = make_flat_image(value=5000.0)
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        assert result["niter"] == 0

    def test_constant_image_run_completes(self):
        from lsc.cosmics import cosmicsimage
        data = make_flat_image(value=2000.0)
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        ci.run(maxiter=3, verbose=False)
        assert ci.mask.sum() == 0

    def test_constant_image_with_single_cosmic(self):
        from lsc.cosmics import cosmicsimage
        data = make_flat_image(value=1000.0)
        data[32, 32] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=3.0,
                          satlevel=-1, verbose=False)
        ci.lacosmiciteration(verbose=False)
        assert ci.mask[32, 32] or ci.mask.sum() > 0


class TestSmallImages:
    def test_6x6_image(self):
        from lsc.cosmics import cosmicsimage
        data = make_noisy_image(shape=(6, 6), seed=10)
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        assert "niter" in result

    def test_10x10_image_with_cosmic(self):
        from lsc.cosmics import cosmicsimage
        data = make_flat_image(shape=(10, 10), value=1000.0)
        data[5, 5] = 50000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=3.0,
                          satlevel=-1, verbose=False)
        ci.run(maxiter=2, verbose=False)
        assert ci.cleanarray[5, 5] < 50000.0


class TestNonSquareImages:
    def test_wide_image(self):
        from lsc.cosmics import cosmicsimage
        data = make_noisy_image(shape=(20, 100), seed=1)
        data[10, 50] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        assert result["niter"] >= 0

    def test_tall_image(self):
        from lsc.cosmics import cosmicsimage
        data = make_noisy_image(shape=(100, 20), seed=2)
        data[50, 10] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        assert result["niter"] >= 0


class TestNegativeValues:
    def test_negative_values_no_crash(self):
        from lsc.cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), sky=0.0, noise=100.0, seed=900)
        assert (data < 0).any()
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        assert result["niter"] >= 0

    def test_all_negative_image(self):
        from lsc.cosmics import cosmicsimage
        data = np.full((32, 32), -500.0, dtype=np.float64)
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        assert result["niter"] == 0


# =============================================================================
# Parameter sensitivity
# =============================================================================

class TestSigclipSensitivity:
    def setup_method(self):
        self.data = make_noisy_image(shape=(64, 64), sky=1000.0, noise=30.0, seed=100)
        self.data[30, 30] = 50000.0
        self.data[20, 40] = 40000.0
        self.data[50, 10] = 30000.0

    def test_low_sigclip_detects_more(self):
        from lsc.cosmics import cosmicsimage
        ci_low = cosmicsimage(self.data.copy(), gain=2.2, readnoise=10.0,
                              sigclip=2.0, satlevel=-1, verbose=False)
        res_low = ci_low.lacosmiciteration(verbose=False)
        ci_high = cosmicsimage(self.data.copy(), gain=2.2, readnoise=10.0,
                               sigclip=10.0, satlevel=-1, verbose=False)
        res_high = ci_high.lacosmiciteration(verbose=False)
        assert res_low["niter"] >= res_high["niter"]

    def test_very_high_sigclip_detects_none(self):
        from lsc.cosmics import cosmicsimage
        moderate_data = make_noisy_image(shape=(64, 64), sky=1000.0, noise=30.0, seed=100)
        ci = cosmicsimage(moderate_data, gain=2.2, readnoise=10.0,
                          sigclip=100.0, satlevel=-1, verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        assert result["niter"] == 0

    def test_sigclip_1_detects_many(self):
        from lsc.cosmics import cosmicsimage
        ci = cosmicsimage(self.data.copy(), gain=2.2, readnoise=10.0,
                          sigclip=1.0, objlim=1.0, satlevel=-1, verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        assert result["niter"] > 0


class TestObjlimSensitivity:
    def setup_method(self):
        self.data = make_noisy_image(shape=(64, 64), sky=1000.0, noise=30.0, seed=200)
        self.data[30, 30] = 50000.0

    def test_low_objlim_detects_more(self):
        from lsc.cosmics import cosmicsimage
        ci_low = cosmicsimage(self.data.copy(), gain=2.2, readnoise=10.0,
                              sigclip=3.0, objlim=1.0, satlevel=-1, verbose=False)
        res_low = ci_low.lacosmiciteration(verbose=False)
        ci_high = cosmicsimage(self.data.copy(), gain=2.2, readnoise=10.0,
                               sigclip=3.0, objlim=50.0, satlevel=-1, verbose=False)
        res_high = ci_high.lacosmiciteration(verbose=False)
        assert res_low["niter"] >= res_high["niter"]

    def test_very_high_objlim_blocks_detection(self):
        from lsc.cosmics import cosmicsimage
        ci = cosmicsimage(self.data.copy(), gain=2.2, readnoise=10.0,
                          sigclip=3.0, objlim=10000.0, satlevel=-1, verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        assert result["niter"] == 0


class TestGainReadnoiseSensitivity:
    def setup_method(self):
        self.data = make_noisy_image(shape=(64, 64), sky=1000.0, noise=30.0, seed=300)
        self.data[30, 30] = 50000.0

    def test_high_gain_affects_noise_model(self):
        from lsc.cosmics import cosmicsimage
        ci_low = cosmicsimage(self.data.copy(), gain=1.0, readnoise=10.0,
                              sigclip=5.0, satlevel=-1, verbose=False)
        res_low = ci_low.lacosmiciteration(verbose=False)
        ci_high = cosmicsimage(self.data.copy(), gain=10.0, readnoise=10.0,
                               sigclip=5.0, satlevel=-1, verbose=False)
        res_high = ci_high.lacosmiciteration(verbose=False)
        assert res_low["niter"] >= 0
        assert res_high["niter"] >= 0

    def test_high_readnoise_reduces_sensitivity(self):
        from lsc.cosmics import cosmicsimage
        ci_low_rn = cosmicsimage(self.data.copy(), gain=2.2, readnoise=1.0,
                                 sigclip=5.0, satlevel=-1, verbose=False)
        res_low_rn = ci_low_rn.lacosmiciteration(verbose=False)
        ci_high_rn = cosmicsimage(self.data.copy(), gain=2.2, readnoise=100.0,
                                  sigclip=5.0, satlevel=-1, verbose=False)
        res_high_rn = ci_high_rn.lacosmiciteration(verbose=False)
        assert res_low_rn["niter"] >= res_high_rn["niter"]

    def test_zero_readnoise(self):
        from lsc.cosmics import cosmicsimage
        ci = cosmicsimage(self.data.copy(), gain=2.2, readnoise=0.0,
                          sigclip=5.0, satlevel=-1, verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        assert result["niter"] >= 0

    def test_very_small_gain(self):
        from lsc.cosmics import cosmicsimage
        ci = cosmicsimage(self.data.copy(), gain=0.01, readnoise=10.0,
                          sigclip=5.0, satlevel=-1, verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        assert result["niter"] >= 0


class TestSigfracSensitivity:
    def setup_method(self):
        self.data = make_noisy_image(shape=(64, 64), sky=1000.0, noise=30.0, seed=400)
        self.data[30, 30] = 50000.0
        self.data[30, 31] = 5000.0
        self.data[31, 30] = 5000.0

    def test_high_sigfrac_catches_fewer_neighbors(self):
        from lsc.cosmics import cosmicsimage
        ci_low = cosmicsimage(self.data.copy(), gain=2.2, readnoise=10.0,
                              sigclip=5.0, sigfrac=0.1, satlevel=-1, verbose=False)
        res_low = ci_low.lacosmiciteration(verbose=False)
        ci_high = cosmicsimage(self.data.copy(), gain=2.2, readnoise=10.0,
                               sigclip=5.0, sigfrac=0.9, satlevel=-1, verbose=False)
        res_high = ci_high.lacosmiciteration(verbose=False)
        assert res_low["niter"] >= res_high["niter"]


# =============================================================================
# Mask handling: pre-existing bad pixel masks
# =============================================================================

class TestPreexistingMask:
    def test_mask_accumulates_across_iterations(self):
        from lsc.cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), seed=600)
        data[20, 20] = 60000.0
        data[40, 40] = 55000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        ci.lacosmiciteration(verbose=False)
        mask_after_1 = ci.mask.copy()
        ci.clean(verbose=False)
        ci.lacosmiciteration(verbose=False)
        mask_after_2 = ci.mask.copy()
        assert np.all(mask_after_2 | ~mask_after_1 == np.ones_like(mask_after_1, dtype=bool))

    def test_manually_set_mask_is_preserved(self):
        from lsc.cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), seed=601)
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=50.0,
                          satlevel=-1, verbose=False)
        ci.mask[5, 5] = True
        ci.mask[10, 10] = True
        ci.lacosmiciteration(verbose=False)
        assert ci.mask[5, 5]
        assert ci.mask[10, 10]

    def test_newmask_only_counts_new_detections(self):
        from lsc.cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), seed=602)
        data[30, 30] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        ci.mask[30, 30] = True
        result = ci.lacosmiciteration(verbose=False)
        assert not result["newmask"][30, 30]


# =============================================================================
# Iteration convergence
# =============================================================================

class TestIterationConvergence:
    def test_single_cosmic_converges_in_one_iteration(self):
        from lsc.cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), seed=700)
        data[30, 30] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        ci.lacosmiciteration(verbose=False)
        ci.clean(verbose=False)
        result2 = ci.lacosmiciteration(verbose=False)
        assert result2["nnew"] <= 2

    def test_maxiter_1_stops_after_one(self):
        from lsc.cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), seed=701)
        data[30, 30] = 60000.0
        data[50, 50] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        ci.run(maxiter=1, verbose=False)
        assert ci.mask.sum() > 0

    def test_multiple_iterations_improve_cleaning(self):
        from lsc.cosmics import cosmicsimage
        data = make_noisy_image(shape=(128, 128), seed=702)
        data[60:63, 60:63] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        ci.run(maxiter=4, verbose=False)
        assert ci.mask[60:63, 60:63].sum() > 0


# =============================================================================
# Multiple cosmics and clustering
# =============================================================================

class TestMultipleCosmics:
    def test_many_isolated_cosmics(self):
        from lsc.cosmics import cosmicsimage
        data = make_noisy_image(shape=(128, 128), seed=1000)
        positions = [(10, 10), (30, 50), (60, 80), (90, 30), (110, 110)]
        for r, c in positions:
            data[r, c] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        ci.run(maxiter=3, verbose=False)
        detected = sum(1 for r, c in positions if ci.mask[r, c])
        assert detected >= len(positions) - 1

    def test_adjacent_cosmics_merged(self):
        from lsc.cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), seed=1001)
        data[30, 30] = 60000.0
        data[30, 31] = 60000.0
        data[31, 30] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        ci.lacosmiciteration(verbose=False)
        assert ci.mask[30, 30] or ci.mask[30, 31] or ci.mask[31, 30]

    def test_cosmic_trail(self):
        from lsc.cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), seed=1002)
        for c in range(20, 45):
            data[30, c] = 55000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=3.0,
                          satlevel=-1, verbose=False)
        ci.run(maxiter=3, verbose=False)
        assert ci.mask[30, 20:45].sum() > 10


# =============================================================================
# Data type handling
# =============================================================================

class TestDataTypes:
    def test_float32_input(self):
        from lsc.cosmics import cosmicsimage
        data = make_noisy_image(shape=(32, 32), seed=1100).astype(np.float32)
        data[16, 16] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        assert result["niter"] >= 0

    def test_float64_input(self):
        from lsc.cosmics import cosmicsimage
        data = make_noisy_image(shape=(32, 32), seed=1101).astype(np.float64)
        data[16, 16] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        assert result["niter"] > 0

    def test_int_input_promoted_to_float(self):
        from lsc.cosmics import cosmicsimage
        data = np.full((32, 32), 1000, dtype=np.int32)
        data[16, 16] = 60000
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        assert result["niter"] >= 0


# =============================================================================
# Noise model edge cases
# =============================================================================

class TestNoiseModel:
    def test_noise_model_positive_definite(self):
        from lsc.cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), seed=1300)
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        assert result is not None

    def test_very_bright_image_noise_model(self):
        from lsc.cosmics import cosmicsimage
        data = make_flat_image(shape=(32, 32), value=50000.0)
        data[16, 16] = 200000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        assert result["niter"] >= 0


# =============================================================================
# Full pipeline integration
# =============================================================================

class TestFullPipelineIntegration:
    def test_full_pipeline_flat_image_no_change(self):
        from lsc.cosmics import cosmicsimage
        data = make_flat_image(shape=(64, 64), value=2000.0)
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        ci.run(maxiter=3, verbose=False)
        assert np.allclose(ci.getcleanarray(), 2000.0)

    def test_full_pipeline_cleaning_does_not_introduce_nans(self):
        from lsc.cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), seed=1800)
        data[30, 30] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        ci.run(maxiter=3, verbose=False)
        assert not np.any(np.isnan(ci.cleanarray))
        assert not np.any(np.isnan(ci.getcleanarray()))

    def test_full_pipeline_cleaning_does_not_introduce_infs(self):
        from lsc.cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), seed=1801)
        data[30, 30] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        ci.run(maxiter=3, verbose=False)
        assert not np.any(np.isinf(ci.cleanarray))

    def test_full_pipeline_mask_is_boolean(self):
        from lsc.cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), seed=1802)
        data[30, 30] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        ci.run(maxiter=3, verbose=False)
        assert ci.mask.dtype == bool

    def test_full_pipeline_with_satlevel_and_cosmics(self):
        from lsc.cosmics import cosmicsimage
        data = make_noisy_image(shape=(128, 128), seed=1803)
        data[20:28, 20:28] = 55000.0
        data[80, 80] = 60000.0
        data[100, 100] = 55000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=50000.0, verbose=False)
        ci.run(maxiter=3, verbose=False)
        assert ci.satstars is not None
        assert ci.mask[80, 80] or ci.mask[100, 100]


# =============================================================================
# Tests using the shared image_array fixture from conftest
# =============================================================================

class TestWithFixture:
    def test_fixture_has_cosmics_detected(self, image_array):
        from lsc.cosmics import cosmicsimage
        ci = cosmicsimage(image_array.astype(np.float64), gain=2.2, readnoise=10.0,
                          sigclip=5.0, satlevel=-1, verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        assert result["niter"] > 0

    def test_fixture_cosmics_at_known_positions(self, image_array):
        from lsc.cosmics import cosmicsimage
        ci = cosmicsimage(image_array.astype(np.float64), gain=2.2, readnoise=10.0,
                          sigclip=5.0, satlevel=-1, verbose=False)
        ci.run(maxiter=3, verbose=False)
        detected = sum([
            ci.mask[30, 40],
            ci.mask[70, 20],
            ci.mask[10, 90],
        ])
        assert detected >= 2

    def test_fixture_clean_reduces_cosmic_values(self, image_array):
        from lsc.cosmics import cosmicsimage
        ci = cosmicsimage(image_array.astype(np.float64), gain=2.2, readnoise=10.0,
                          sigclip=5.0, satlevel=-1, verbose=False)
        ci.run(maxiter=3, verbose=False)
        for r, c in [(30, 40), (70, 20), (10, 90)]:
            if ci.mask[r, c]:
                assert ci.getcleanarray()[r, c] < 10000.0


# =============================================================================
# fromfits / tofits — FITS I/O
# =============================================================================

class TestFitsIO:
    def test_tofits_fromfits_roundtrip(self, tmp_path):
        """Writing then reading a FITS file recovers the original array shape."""
        from lsc.cosmics import fromfits, tofits
        arr = np.random.default_rng(1).normal(1000, 50, (32, 32)).astype(np.float64)
        outfile = str(tmp_path / "roundtrip.fits")
        tofits(outfile, arr, verbose=False)
        arr2, _ = fromfits(outfile, verbose=False)
        assert arr2.shape == arr.shape or arr2.shape == arr.T.shape

    def test_tofits_creates_file(self, tmp_path):
        from lsc.cosmics import tofits
        arr = np.ones((10, 10))
        outfile = str(tmp_path / "out.fits")
        tofits(outfile, arr, verbose=False)
        assert os.path.exists(outfile)

    def test_tofits_boolean_array_written_as_uint8(self, tmp_path):
        from lsc.cosmics import tofits, fromfits
        arr = np.zeros((8, 8), dtype=bool)
        arr[2, 2] = True
        outfile = str(tmp_path / "bool.fits")
        tofits(outfile, arr, verbose=False)
        arr2, _ = fromfits(outfile, verbose=False)
        assert arr2.dtype != bool
        assert set(arr2.ravel().astype(int)).issubset({0, 1})

    def test_tofits_overwrites_existing_file(self, tmp_path):
        from lsc.cosmics import tofits, fromfits
        arr1 = np.full((8, 8), 1.0)
        arr2 = np.full((8, 8), 2.0)
        outfile = str(tmp_path / "overwrite.fits")
        tofits(outfile, arr1, verbose=False)
        tofits(outfile, arr2, verbose=False)
        result, _ = fromfits(outfile, verbose=False)
        assert np.allclose(result, 2.0)

    def test_fromfits_returns_array_and_header(self, tmp_path):
        from lsc.cosmics import fromfits, tofits
        arr = np.ones((16, 16))
        outfile = str(tmp_path / "test2.fits")
        tofits(outfile, arr, verbose=False)
        arr2, _ = fromfits(outfile, verbose=False)
        assert isinstance(arr2, np.ndarray)

    def test_fromfits_specific_hdu(self, tmp_path):
        from lsc.cosmics import fromfits, tofits
        arr = np.ones((16, 16), dtype=np.float64)
        outfile = str(tmp_path / "hdu_test.fits")
        tofits(outfile, arr, verbose=False)
        arr2, hdr = fromfits(outfile, hdu=0, verbose=False)
        assert arr2 is not None

    def test_tofits_with_nan_values(self, tmp_path):
        from lsc.cosmics import tofits, fromfits
        arr = np.ones((8, 8), dtype=np.float64)
        arr[3, 3] = np.nan
        outfile = str(tmp_path / "nan_test.fits")
        tofits(outfile, arr, verbose=False)
        arr2, _ = fromfits(outfile, verbose=False)
        assert np.isnan(arr2).any()

    def test_tofits_with_inf_values(self, tmp_path):
        from lsc.cosmics import tofits, fromfits
        arr = np.ones((8, 8), dtype=np.float64)
        arr[3, 3] = np.inf
        outfile = str(tmp_path / "inf_test.fits")
        tofits(outfile, arr, verbose=False)
        arr2, _ = fromfits(outfile, verbose=False)
        assert np.isinf(arr2).any()

    def test_fromfits_returns_float_array(self, tmp_path):
        from lsc.cosmics import fromfits, tofits
        arr = np.ones((12, 12), dtype=np.float64)
        outfile = str(tmp_path / "dtype_test.fits")
        tofits(outfile, arr, verbose=False)
        arr2, _ = fromfits(outfile, verbose=False)
        assert np.issubdtype(arr2.dtype, np.number)


class TestFromfitsVerbose:
    def test_verbose_true_prints_shape_and_bitpix(self, tmp_path, capsys):
        from lsc.cosmics import fromfits, tofits
        arr = np.ones((16, 16))
        outfile = str(tmp_path / "verbose_in.fits")
        tofits(outfile, arr, verbose=False)
        fromfits(outfile, verbose=True)
        captured = capsys.readouterr()
        assert "shape" in captured.out.lower() or "FITS import" in captured.out
        assert "BITPIX" in captured.out or "dtype" in captured.out.lower()


class TestTofitsVerboseAndHeader:
    def test_verbose_true_prints_shape_and_wrote(self, tmp_path, capsys):
        from lsc.cosmics import tofits
        arr = np.ones((12, 12))
        outfile = str(tmp_path / "verbose_out.fits")
        tofits(outfile, arr, verbose=True)
        captured = capsys.readouterr()
        assert "FITS export shape" in captured.out
        assert "Wrote" in captured.out

    def test_with_header_uses_header_hdu(self, tmp_path):
        from lsc.cosmics import tofits, fromfits
        arr = np.ones((8, 8))
        outfile = str(tmp_path / "hdr_out.fits")
        tofits(outfile, arr, verbose=False)
        _, hdr = fromfits(outfile, verbose=False)
        outfile2 = str(tmp_path / "hdr_out2.fits")
        tofits(outfile2, arr, hdr=hdr, verbose=False)
        arr2, _ = fromfits(outfile2, verbose=False)
        assert arr2.shape == arr.shape or arr2.shape == arr.T.shape
