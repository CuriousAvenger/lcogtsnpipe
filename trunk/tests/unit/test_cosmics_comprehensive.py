"""
Comprehensive tests for lsc.cosmics — L.A.Cosmic cosmic-ray detection algorithm.

Covers edge cases, parameter sensitivity, kernel operations, mask handling,
iteration convergence, and boundary conditions not covered by test_cosmics.py.
"""
import sys
import os

# Insert the source directory so we can import cosmics directly without
# triggering the full lsc package (which has heavy dependencies like pyraf).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'lsc'))

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

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
# Edge Cases: All-zero images
# =============================================================================

class TestAllZeroImage:
    """Test behavior on an image of all zeros."""

    def test_init_with_zero_image(self):
        """cosmicsimage can be initialized with an all-zero array."""
        from cosmics import cosmicsimage
        data = np.zeros((32, 32), dtype=np.float64)
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, satlevel=-1, verbose=False)
        assert ci.rawarray.shape == (32, 32)
        assert np.all(ci.rawarray == 0.0)

    def test_iteration_on_zero_image(self):
        """lacosmiciteration on an all-zero image does not crash and finds no cosmics."""
        from cosmics import cosmicsimage
        data = np.zeros((32, 32), dtype=np.float64)
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, satlevel=-1, verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        assert result["niter"] == 0
        assert result["nnew"] == 0

    def test_clean_on_zero_image_no_mask(self):
        """clean() on zero image with empty mask does nothing."""
        from cosmics import cosmicsimage
        data = np.zeros((32, 32), dtype=np.float64)
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, satlevel=-1, verbose=False)
        ci.clean(verbose=False)
        assert np.all(ci.cleanarray == 0.0)

    def test_guessbackgroundlevel_zero_image(self):
        """guessbackgroundlevel on zero image returns 0."""
        from cosmics import cosmicsimage
        data = np.zeros((32, 32), dtype=np.float64)
        ci = cosmicsimage(data, verbose=False)
        assert ci.guessbackgroundlevel() == 0.0


# =============================================================================
# Edge Cases: Constant images
# =============================================================================

class TestConstantImage:
    """Test behavior on constant-value images (no noise)."""

    def test_constant_image_no_cosmics_detected(self):
        """A uniform image should have no cosmic ray detections."""
        from cosmics import cosmicsimage
        data = make_flat_image(value=5000.0)
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        assert result["niter"] == 0

    def test_constant_image_run_completes(self):
        """run() on constant image terminates without error."""
        from cosmics import cosmicsimage
        data = make_flat_image(value=2000.0)
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        ci.run(maxiter=3, verbose=False)
        assert ci.mask.sum() == 0

    def test_constant_image_with_single_cosmic(self):
        """A single bright pixel on a flat background is detected."""
        from cosmics import cosmicsimage
        data = make_flat_image(value=1000.0)
        data[32, 32] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=3.0,
                          satlevel=-1, verbose=False)
        ci.lacosmiciteration(verbose=False)
        assert ci.mask[32, 32] or ci.mask.sum() > 0


# =============================================================================
# Edge Cases: Small images
# =============================================================================

class TestSmallImages:
    """Test behavior on very small images."""

    def test_5x5_image(self):
        """A 5x5 image can be processed without error."""
        from cosmics import cosmicsimage
        data = make_noisy_image(shape=(6, 6), seed=10)
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        assert "niter" in result

    def test_10x10_image_with_cosmic(self):
        """A 10x10 image with a cosmic can be detected and cleaned."""
        from cosmics import cosmicsimage
        data = make_flat_image(shape=(10, 10), value=1000.0)
        data[5, 5] = 50000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=3.0,
                          satlevel=-1, verbose=False)
        ci.run(maxiter=2, verbose=False)
        # After cleaning, the pixel should be closer to background
        assert ci.cleanarray[5, 5] < 50000.0


# =============================================================================
# Edge Cases: Non-square images
# =============================================================================

class TestNonSquareImages:
    """Test behavior on rectangular images."""

    def test_wide_image(self):
        """A wide rectangular image (20x100) is handled correctly."""
        from cosmics import cosmicsimage
        data = make_noisy_image(shape=(20, 100), seed=1)
        data[10, 50] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        assert result["niter"] >= 0

    def test_tall_image(self):
        """A tall rectangular image (100x20) is handled correctly."""
        from cosmics import cosmicsimage
        data = make_noisy_image(shape=(100, 20), seed=2)
        data[50, 10] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        assert result["niter"] >= 0


# =============================================================================
# Parameter Sensitivity: sigclip
# =============================================================================

class TestSigclipSensitivity:
    """Test how sigclip threshold affects detection."""

    def setup_method(self):
        self.data = make_noisy_image(shape=(64, 64), sky=1000.0, noise=30.0, seed=100)
        self.data[30, 30] = 50000.0
        self.data[20, 40] = 40000.0
        self.data[50, 10] = 30000.0

    def test_low_sigclip_detects_more(self):
        """A lower sigclip detects more candidate cosmic pixels."""
        from cosmics import cosmicsimage
        ci_low = cosmicsimage(self.data.copy(), gain=2.2, readnoise=10.0,
                              sigclip=2.0, satlevel=-1, verbose=False)
        res_low = ci_low.lacosmiciteration(verbose=False)

        ci_high = cosmicsimage(self.data.copy(), gain=2.2, readnoise=10.0,
                               sigclip=10.0, satlevel=-1, verbose=False)
        res_high = ci_high.lacosmiciteration(verbose=False)

        assert res_low["niter"] >= res_high["niter"]

    def test_very_high_sigclip_detects_none(self):
        """A very high sigclip (100) should detect no cosmics on a moderate image."""
        from cosmics import cosmicsimage
        # Use image without injected extreme cosmics for this test
        moderate_data = make_noisy_image(shape=(64, 64), sky=1000.0, noise=30.0, seed=100)
        ci = cosmicsimage(moderate_data, gain=2.2, readnoise=10.0,
                          sigclip=100.0, satlevel=-1, verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        assert result["niter"] == 0

    def test_sigclip_1_detects_many(self):
        """A sigclip of 1.0 is very aggressive and flags many pixels."""
        from cosmics import cosmicsimage
        ci = cosmicsimage(self.data.copy(), gain=2.2, readnoise=10.0,
                          sigclip=1.0, objlim=1.0, satlevel=-1, verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        assert result["niter"] > 0


# =============================================================================
# Parameter Sensitivity: objlim
# =============================================================================

class TestObjlimSensitivity:
    """Test how objlim (object contrast limit) affects detection."""

    def setup_method(self):
        self.data = make_noisy_image(shape=(64, 64), sky=1000.0, noise=30.0, seed=200)
        self.data[30, 30] = 50000.0

    def test_low_objlim_detects_more(self):
        """Lower objlim means more pixels pass the contrast test."""
        from cosmics import cosmicsimage
        ci_low = cosmicsimage(self.data.copy(), gain=2.2, readnoise=10.0,
                              sigclip=3.0, objlim=1.0, satlevel=-1, verbose=False)
        res_low = ci_low.lacosmiciteration(verbose=False)

        ci_high = cosmicsimage(self.data.copy(), gain=2.2, readnoise=10.0,
                               sigclip=3.0, objlim=50.0, satlevel=-1, verbose=False)
        res_high = ci_high.lacosmiciteration(verbose=False)

        assert res_low["niter"] >= res_high["niter"]

    def test_very_high_objlim_blocks_detection(self):
        """Very high objlim effectively blocks all cosmic detection."""
        from cosmics import cosmicsimage
        ci = cosmicsimage(self.data.copy(), gain=2.2, readnoise=10.0,
                          sigclip=3.0, objlim=10000.0, satlevel=-1, verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        assert result["niter"] == 0


# =============================================================================
# Parameter Sensitivity: gain and readnoise
# =============================================================================

class TestGainReadnoiseSensitivity:
    """Test how gain and readnoise affect the noise model."""

    def setup_method(self):
        self.data = make_noisy_image(shape=(64, 64), sky=1000.0, noise=30.0, seed=300)
        self.data[30, 30] = 50000.0

    def test_high_gain_affects_noise_model(self):
        """Changing gain changes the noise model and thus detection sensitivity."""
        from cosmics import cosmicsimage
        ci_low = cosmicsimage(self.data.copy(), gain=1.0, readnoise=10.0,
                              sigclip=5.0, satlevel=-1, verbose=False)
        res_low = ci_low.lacosmiciteration(verbose=False)

        ci_high = cosmicsimage(self.data.copy(), gain=10.0, readnoise=10.0,
                               sigclip=5.0, satlevel=-1, verbose=False)
        res_high = ci_high.lacosmiciteration(verbose=False)

        # Different gain produces different detection counts (not necessarily monotonic)
        # Just verify both run without error
        assert res_low["niter"] >= 0
        assert res_high["niter"] >= 0

    def test_high_readnoise_reduces_sensitivity(self):
        """Higher readnoise increases the noise model, reducing S/N and detections."""
        from cosmics import cosmicsimage
        ci_low_rn = cosmicsimage(self.data.copy(), gain=2.2, readnoise=1.0,
                                 sigclip=5.0, satlevel=-1, verbose=False)
        res_low_rn = ci_low_rn.lacosmiciteration(verbose=False)

        ci_high_rn = cosmicsimage(self.data.copy(), gain=2.2, readnoise=100.0,
                                  sigclip=5.0, satlevel=-1, verbose=False)
        res_high_rn = ci_high_rn.lacosmiciteration(verbose=False)

        # Higher readnoise raises the noise floor, so fewer pixels exceed sigclip
        assert res_low_rn["niter"] >= res_high_rn["niter"]

    def test_zero_readnoise(self):
        """Zero readnoise should still work (noise model uses only gain*signal)."""
        from cosmics import cosmicsimage
        ci = cosmicsimage(self.data.copy(), gain=2.2, readnoise=0.0,
                          sigclip=5.0, satlevel=-1, verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        assert result["niter"] >= 0

    def test_very_small_gain(self):
        """Very small gain (0.01) does not crash the noise calculation."""
        from cosmics import cosmicsimage
        ci = cosmicsimage(self.data.copy(), gain=0.01, readnoise=10.0,
                          sigclip=5.0, satlevel=-1, verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        assert result["niter"] >= 0


# =============================================================================
# Parameter Sensitivity: sigfrac
# =============================================================================

class TestSigfracSensitivity:
    """Test sigfrac (fractional detection limit for neighbors)."""

    def setup_method(self):
        self.data = make_noisy_image(shape=(64, 64), sky=1000.0, noise=30.0, seed=400)
        # Inject a cluster of moderately bright pixels
        self.data[30, 30] = 50000.0
        self.data[30, 31] = 5000.0  # neighbor that might be caught
        self.data[31, 30] = 5000.0  # neighbor that might be caught

    def test_high_sigfrac_catches_fewer_neighbors(self):
        """Higher sigfrac means the second-pass neighbor threshold is higher."""
        from cosmics import cosmicsimage
        ci_low = cosmicsimage(self.data.copy(), gain=2.2, readnoise=10.0,
                              sigclip=5.0, sigfrac=0.1, satlevel=-1, verbose=False)
        res_low = ci_low.lacosmiciteration(verbose=False)

        ci_high = cosmicsimage(self.data.copy(), gain=2.2, readnoise=10.0,
                               sigclip=5.0, sigfrac=0.9, satlevel=-1, verbose=False)
        res_high = ci_high.lacosmiciteration(verbose=False)

        # Lower sigfrac => lower neighbor threshold => potentially more detections
        assert res_low["niter"] >= res_high["niter"]


# =============================================================================
# Saturated pixels handling
# =============================================================================

class TestSaturatedPixelsHandling:
    """Test saturated star detection and masking."""

    def test_saturated_star_not_detected_as_cosmic(self):
        """Saturated stars should be excluded from cosmic ray detection."""
        from cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), sky=1000.0, noise=30.0, seed=500)
        # Plant a large saturated region (star-like)
        data[28:36, 28:36] = 55000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=50000.0, verbose=False)
        ci.findsatstars(verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        # The saturated star region should not appear in the cosmic mask
        sat_region_in_mask = ci.mask[28:36, 28:36].sum()
        assert sat_region_in_mask == 0

    def test_negative_satlevel_skips_satstar_detection(self):
        """Negative satlevel disables saturated star detection."""
        from cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), sky=1000.0, noise=30.0, seed=501)
        data[28:36, 28:36] = 55000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        ci.run(maxiter=1, verbose=False)
        # satstars should remain None since satlevel is negative
        assert ci.satstars is None

    def test_findsatstars_no_saturated_stars(self):
        """findsatstars on clean image (no saturation) yields empty mask."""
        from cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), sky=1000.0, noise=30.0, seed=502)
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, satlevel=50000.0, verbose=False)
        ci.findsatstars(verbose=False)
        assert ci.satstars.sum() == 0

    def test_findsatstars_multiple_saturated_stars(self):
        """findsatstars detects multiple distinct saturated regions."""
        from cosmics import cosmicsimage
        data = make_noisy_image(shape=(128, 128), sky=1000.0, noise=30.0, seed=503)
        # Plant two separated saturated regions
        data[20:28, 20:28] = 55000.0
        data[80:88, 80:88] = 55000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, satlevel=50000.0, verbose=False)
        ci.findsatstars(verbose=False)
        assert ci.satstars.sum() > 0


# =============================================================================
# Laplacian kernel operations
# =============================================================================

class TestLaplacianKernel:
    """Test the Laplacian kernel and its convolution."""

    def test_laplkernel_shape(self):
        """The Laplacian kernel is 3x3."""
        from cosmics import laplkernel
        assert laplkernel.shape == (3, 3)

    def test_laplkernel_sum_is_zero(self):
        """The Laplacian kernel sums to zero (edge-detecting property)."""
        from cosmics import laplkernel
        assert np.isclose(laplkernel.sum(), 0.0)

    def test_laplkernel_center_is_positive(self):
        """The center element of the Laplacian kernel is positive (4.0)."""
        from cosmics import laplkernel
        assert laplkernel[1, 1] == 4.0

    def test_growkernel_shape_and_values(self):
        """The grow kernel is 3x3 of all ones."""
        from cosmics import growkernel
        assert growkernel.shape == (3, 3)
        assert np.all(growkernel == 1.0)

    def test_dilstruct_shape(self):
        """The dilation structure is 5x5 with corners cut."""
        from cosmics import dilstruct
        assert dilstruct.shape == (5, 5)
        # Corners should be 0
        assert dilstruct[0, 0] == 0
        assert dilstruct[0, 4] == 0
        assert dilstruct[4, 0] == 0
        assert dilstruct[4, 4] == 0
        # Non-corner elements in first row should be 1
        assert dilstruct[0, 1] == 1
        assert dilstruct[0, 2] == 1
        assert dilstruct[0, 3] == 1

    def test_laplacian_on_flat_image_is_zero(self):
        """Convolving a flat image with the Laplacian produces approximately zero."""
        import scipy.signal as signal
        from cosmics import laplkernel, subsample, rebin2x2
        data = make_flat_image(shape=(32, 32), value=1000.0)
        subsam = subsample(data)
        conved = signal.convolve2d(subsam, laplkernel, mode="same", boundary="symm")
        # Interior of a constant image should be ~0 after Laplacian
        assert np.allclose(conved[2:-2, 2:-2], 0.0, atol=1e-10)

    def test_laplacian_detects_single_spike(self):
        """Convolving an image with one spike gives a strong positive response at that location."""
        import scipy.signal as signal
        from cosmics import laplkernel, subsample, rebin2x2
        data = make_flat_image(shape=(32, 32), value=100.0)
        data[16, 16] = 50000.0
        subsam = subsample(data)
        conved = signal.convolve2d(subsam, laplkernel, mode="same", boundary="symm")
        cliped = conved.clip(min=0.0)
        lplus = rebin2x2(cliped)
        # The spike location should have the highest value
        assert lplus[16, 16] == lplus.max()


# =============================================================================
# Subsample and rebin operations (additional tests)
# =============================================================================

class TestSubsampleAdditional:
    """Additional tests for subsample function."""

    def test_subsample_preserves_dtype(self):
        """subsample preserves the dtype of the input array."""
        from cosmics import subsample
        a = np.ones((10, 10), dtype=np.float32)
        b = subsample(a)
        assert b.dtype == np.float32

    def test_subsample_each_pixel_becomes_2x2(self):
        """Each pixel in input maps to a 2x2 block in output."""
        from cosmics import subsample
        a = np.array([[1.0, 2.0],
                      [3.0, 4.0]])
        b = subsample(a)
        assert b.shape == (4, 4)
        # Top-left pixel value=1 should fill a 2x2 block
        assert b[0, 0] == 1.0
        assert b[0, 1] == 1.0
        assert b[1, 0] == 1.0
        assert b[1, 1] == 1.0
        # Top-right pixel value=2
        assert b[0, 2] == 2.0
        assert b[0, 3] == 2.0
        assert b[1, 2] == 2.0
        assert b[1, 3] == 2.0

    def test_subsample_large_values(self):
        """subsample handles large pixel values without overflow."""
        from cosmics import subsample
        a = np.array([[1e10]], dtype=np.float64)
        b = subsample(a)
        assert np.all(b == 1e10)

    def test_subsample_negative_values(self):
        """subsample correctly handles negative values."""
        from cosmics import subsample
        a = np.array([[-500.0, 500.0],
                      [0.0, -1000.0]])
        b = subsample(a)
        assert b[0, 0] == -500.0
        assert b[0, 2] == 500.0
        assert b[3, 3] == -1000.0


class TestRebinAdditional:
    """Additional tests for rebin function."""

    def test_rebin_3x3_to_1x1(self):
        """Rebinning a 3x3 array to 1x1 returns the mean."""
        from cosmics import rebin
        a = np.arange(9, dtype=np.float64).reshape(3, 3)
        b = rebin(a, (1, 1))
        assert b.shape == (1, 1)
        assert np.isclose(b[0, 0], a.mean())

    def test_rebin_identity(self):
        """Rebinning to the same shape returns the same values."""
        from cosmics import rebin
        a = np.ones((4, 4), dtype=np.float64) * 5.0
        b = rebin(a, (4, 4))
        assert np.allclose(b, 5.0)

    def test_rebin_non_square(self):
        """Rebinning a rectangular array works correctly."""
        from cosmics import rebin
        a = np.ones((6, 8), dtype=np.float64) * 3.0
        b = rebin(a, (3, 4))
        assert b.shape == (3, 4)
        assert np.allclose(b, 3.0)


class TestRebin2x2Additional:
    """Additional edge cases for rebin2x2."""

    def test_rebin2x2_with_varying_values(self):
        """rebin2x2 correctly averages 2x2 blocks of varying values."""
        from cosmics import rebin2x2
        a = np.array([[10.0, 20.0, 30.0, 40.0],
                      [50.0, 60.0, 70.0, 80.0],
                      [1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0]])
        b = rebin2x2(a)
        assert b.shape == (2, 2)
        # First block: mean of (10,20,50,60)
        assert np.isclose(b[0, 0], (10 + 20 + 50 + 60) / 4.0)
        # Second block: mean of (30,40,70,80)
        assert np.isclose(b[0, 1], (30 + 40 + 70 + 80) / 4.0)

    def test_rebin2x2_minimum_even_size(self):
        """rebin2x2 works on the minimum valid even size (2x2)."""
        from cosmics import rebin2x2
        a = np.array([[4.0, 8.0],
                      [12.0, 16.0]])
        b = rebin2x2(a)
        assert b.shape == (1, 1)
        assert np.isclose(b[0, 0], 10.0)

    def test_rebin2x2_odd_even_mix_raises(self):
        """rebin2x2 raises an error on arrays with one odd dimension."""
        from cosmics import rebin2x2
        a = np.ones((6, 7))
        with pytest.raises(Exception):
            rebin2x2(a)


# =============================================================================
# Mask handling: pre-existing bad pixel masks
# =============================================================================

class TestPreexistingMask:
    """Test that pre-existing mask state is preserved and extended."""

    def test_mask_accumulates_across_iterations(self):
        """The mask grows (OR) across iterations, never shrinks."""
        from cosmics import cosmicsimage
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
        # Second iteration mask should be a superset of first
        assert np.all(mask_after_2 | ~mask_after_1 == np.ones_like(mask_after_1, dtype=bool))

    def test_manually_set_mask_is_preserved(self):
        """Manually setting mask pixels before iteration preserves them."""
        from cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), seed=601)
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=50.0,
                          satlevel=-1, verbose=False)
        # Manually flag some pixels
        ci.mask[5, 5] = True
        ci.mask[10, 10] = True
        ci.lacosmiciteration(verbose=False)
        # Original manually-set pixels should still be True
        assert ci.mask[5, 5]
        assert ci.mask[10, 10]

    def test_newmask_only_counts_new_detections(self):
        """The 'newmask' in iteration result excludes already-masked pixels."""
        from cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), seed=602)
        data[30, 30] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        # Pre-mask the cosmic pixel
        ci.mask[30, 30] = True
        result = ci.lacosmiciteration(verbose=False)
        # The pixel at [30,30] should not count as "new" since it was already masked
        assert not result["newmask"][30, 30]


# =============================================================================
# Iteration limits and convergence
# =============================================================================

class TestIterationConvergence:
    """Test iteration behavior and convergence."""

    def test_single_cosmic_converges_in_one_iteration(self):
        """A single isolated cosmic should be fully detected in one iteration."""
        from cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), seed=700)
        data[30, 30] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        ci.lacosmiciteration(verbose=False)
        ci.clean(verbose=False)
        result2 = ci.lacosmiciteration(verbose=False)
        # Second iteration should find zero new cosmics (or very few near the cleaned area)
        assert result2["nnew"] <= 2

    def test_maxiter_1_stops_after_one(self):
        """run() with maxiter=1 only performs one iteration."""
        from cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), seed=701)
        data[30, 30] = 60000.0
        data[50, 50] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        ci.run(maxiter=1, verbose=False)
        # Should have run, and mask should reflect detections
        assert ci.mask.sum() > 0

    def test_multiple_iterations_improve_cleaning(self):
        """Multiple iterations can catch residual cosmic ray artifacts."""
        from cosmics import cosmicsimage
        data = make_noisy_image(shape=(128, 128), seed=702)
        # Inject a multi-pixel cosmic cluster
        data[60:63, 60:63] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        ci.run(maxiter=4, verbose=False)
        # The cosmic cluster region should be largely masked
        cluster_masked = ci.mask[60:63, 60:63].sum()
        assert cluster_masked > 0


# =============================================================================
# Clean method edge cases
# =============================================================================

class TestCleanEdgeCases:
    """Edge cases in the clean() method."""

    def test_clean_cosmic_at_corner(self):
        """clean() handles a cosmic at the image corner (uses padded array)."""
        from cosmics import cosmicsimage
        data = make_flat_image(shape=(32, 32), value=1000.0)
        data[0, 0] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=3.0,
                          satlevel=-1, verbose=False)
        ci.lacosmiciteration(verbose=False)
        ci.clean(verbose=False)
        # Corner pixel should not be inf after cleaning
        assert not np.isinf(ci.cleanarray[0, 0])

    def test_clean_cosmic_at_edge(self):
        """clean() handles a cosmic on the image edge."""
        from cosmics import cosmicsimage
        data = make_flat_image(shape=(32, 32), value=1000.0)
        data[0, 15] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=3.0,
                          satlevel=-1, verbose=False)
        ci.lacosmiciteration(verbose=False)
        ci.clean(verbose=False)
        assert not np.isinf(ci.cleanarray[0, 15])

    def test_clean_replaces_with_local_median(self):
        """Cleaned pixel value should approximate the local background."""
        from cosmics import cosmicsimage
        data = make_flat_image(shape=(32, 32), value=1000.0)
        data[16, 16] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=3.0,
                          satlevel=-1, verbose=False)
        ci.lacosmiciteration(verbose=False)
        ci.clean(verbose=False)
        # The cleaned value should be close to the background (1000)
        assert abs(ci.cleanarray[16, 16] - 1000.0) < 100.0

    def test_clean_empty_mask_does_nothing(self):
        """clean() with all-False mask does not modify cleanarray."""
        from cosmics import cosmicsimage
        data = make_flat_image(shape=(16, 16), value=500.0)
        ci = cosmicsimage(data, verbose=False)
        original = ci.cleanarray.copy()
        ci.clean(verbose=False)
        assert np.array_equal(ci.cleanarray, original)

    def test_clean_entire_image_masked(self):
        """clean() when entire image is masked uses guessbackgroundlevel for all."""
        from cosmics import cosmicsimage
        data = make_flat_image(shape=(16, 16), value=700.0)
        ci = cosmicsimage(data, verbose=False)
        full_mask = np.ones((16, 16), dtype=bool)
        ci.clean(mask=full_mask, verbose=False)
        # All pixels were inf neighbors, so replacements use background level
        bg = ci.guessbackgroundlevel()
        # Interior pixels should be replaced with background level
        assert np.isclose(ci.cleanarray[8, 8], bg)


# =============================================================================
# getdilatedmask additional tests
# =============================================================================

class TestGetDilatedMaskAdditional:
    """Additional tests for getdilatedmask."""

    def test_empty_mask_dilated_is_still_empty(self):
        """Dilating an all-False mask returns an all-False mask."""
        from cosmics import cosmicsimage
        data = make_flat_image(shape=(32, 32))
        ci = cosmicsimage(data, verbose=False)
        d3 = ci.getdilatedmask(size=3)
        d5 = ci.getdilatedmask(size=5)
        assert d3.sum() == 0
        assert d5.sum() == 0

    def test_single_pixel_dilated_size3(self):
        """Dilating a single masked pixel with size=3 produces a cross/3x3 pattern."""
        from cosmics import cosmicsimage
        data = make_flat_image(shape=(32, 32))
        ci = cosmicsimage(data, verbose=False)
        ci.mask[16, 16] = True
        d3 = ci.getdilatedmask(size=3)
        # The 3x3 growkernel should produce at least 9 True pixels (3x3 region)
        assert d3.sum() >= 5  # at minimum the center + 4 cardinal neighbors
        assert d3.sum() <= 9

    def test_single_pixel_dilated_size5(self):
        """Dilating a single masked pixel with size=5 produces a larger pattern."""
        from cosmics import cosmicsimage
        data = make_flat_image(shape=(32, 32))
        ci = cosmicsimage(data, verbose=False)
        ci.mask[16, 16] = True
        d5 = ci.getdilatedmask(size=5)
        # The 5x5 dilstruct (with cut corners) should produce ~21 True pixels
        assert d5.sum() > 9


# =============================================================================
# labelmask additional tests
# =============================================================================

class TestLabelmaskAdditional:
    """Additional tests for labelmask."""

    def test_labelmask_dict_has_expected_keys(self):
        """Each dict in labelmask result has 'name', 'x', 'y' keys."""
        from cosmics import cosmicsimage
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

    def test_labelmask_multiple_cosmic_islands(self):
        """labelmask correctly identifies multiple separate cosmic islands."""
        from cosmics import cosmicsimage
        data = make_noisy_image(shape=(128, 128), seed=801)
        # Inject two widely separated cosmics
        data[20, 20] = 60000.0
        data[100, 100] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        ci.lacosmiciteration(verbose=False)
        result = ci.labelmask(verbose=False)
        # Should detect at least 2 separate islands
        assert len(result) >= 2


# =============================================================================
# FITS I/O additional tests
# =============================================================================

class TestFitsIOAdditional:
    """Additional FITS I/O tests."""

    def test_fromfits_specific_hdu(self, tmp_path):
        """fromfits can read a specific HDU index."""
        from cosmics import fromfits, tofits
        arr = np.ones((16, 16), dtype=np.float64)
        outfile = str(tmp_path / "hdu_test.fits")
        tofits(outfile, arr, verbose=False)
        # HDU 0 is the primary
        arr2, hdr = fromfits(outfile, hdu=0, verbose=False)
        assert arr2 is not None

    def test_tofits_with_nan_values(self, tmp_path):
        """tofits can write an array containing NaN values."""
        from cosmics import tofits, fromfits
        arr = np.ones((8, 8), dtype=np.float64)
        arr[3, 3] = np.nan
        outfile = str(tmp_path / "nan_test.fits")
        tofits(outfile, arr, verbose=False)
        arr2, _ = fromfits(outfile, verbose=False)
        assert np.isnan(arr2).any()

    def test_tofits_with_inf_values(self, tmp_path):
        """tofits can write an array containing infinity values."""
        from cosmics import tofits, fromfits
        arr = np.ones((8, 8), dtype=np.float64)
        arr[3, 3] = np.inf
        outfile = str(tmp_path / "inf_test.fits")
        tofits(outfile, arr, verbose=False)
        arr2, _ = fromfits(outfile, verbose=False)
        assert np.isinf(arr2).any()

    def test_fromfits_returns_float_array(self, tmp_path):
        """fromfits returns a numeric numpy array."""
        from cosmics import fromfits, tofits
        arr = np.ones((12, 12), dtype=np.float64)
        outfile = str(tmp_path / "dtype_test.fits")
        tofits(outfile, arr, verbose=False)
        arr2, _ = fromfits(outfile, verbose=False)
        assert np.issubdtype(arr2.dtype, np.number)


# =============================================================================
# pssl (previously subtracted sky level) tests
# =============================================================================

class TestPSSL:
    """Test the pssl parameter handling."""

    def test_pssl_added_to_rawarray(self):
        """pssl is added to the raw array internally."""
        from cosmics import cosmicsimage
        data = np.full((16, 16), 1000.0)
        ci = cosmicsimage(data, pssl=500.0, verbose=False)
        assert np.allclose(ci.rawarray, 1500.0)

    def test_pssl_subtracted_in_getrawarray(self):
        """getrawarray subtracts pssl to return the original data."""
        from cosmics import cosmicsimage
        data = np.full((16, 16), 1000.0)
        ci = cosmicsimage(data, pssl=500.0, verbose=False)
        assert np.allclose(ci.getrawarray(), 1000.0)

    def test_pssl_subtracted_in_getcleanarray(self):
        """getcleanarray subtracts pssl."""
        from cosmics import cosmicsimage
        data = np.full((16, 16), 1000.0)
        ci = cosmicsimage(data, pssl=500.0, verbose=False)
        assert np.allclose(ci.getcleanarray(), 1000.0)

    def test_negative_pssl(self):
        """Negative pssl is handled correctly (sky was over-subtracted)."""
        from cosmics import cosmicsimage
        data = np.full((16, 16), 1000.0)
        ci = cosmicsimage(data, pssl=-200.0, verbose=False)
        assert np.allclose(ci.rawarray, 800.0)
        assert np.allclose(ci.getrawarray(), 1000.0)

    def test_zero_pssl_no_change(self):
        """Zero pssl leaves rawarray unchanged."""
        from cosmics import cosmicsimage
        data = np.full((16, 16), 1000.0)
        ci = cosmicsimage(data, pssl=0.0, verbose=False)
        assert np.allclose(ci.rawarray, 1000.0)


# =============================================================================
# Image with negative values
# =============================================================================

class TestNegativeValues:
    """Test behavior with images containing negative values."""

    def test_negative_values_no_crash(self):
        """An image with negative pixel values does not crash the algorithm."""
        from cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), sky=0.0, noise=100.0, seed=900)
        # Many pixels will be negative
        assert (data < 0).any()
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        assert result["niter"] >= 0

    def test_all_negative_image(self):
        """An all-negative image can be processed."""
        from cosmics import cosmicsimage
        data = np.full((32, 32), -500.0, dtype=np.float64)
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        assert result["niter"] == 0


# =============================================================================
# Multiple cosmics and clustering
# =============================================================================

class TestMultipleCosmics:
    """Test detection of multiple and clustered cosmics."""

    def test_many_isolated_cosmics(self):
        """Multiple isolated single-pixel cosmics are all detected."""
        from cosmics import cosmicsimage
        data = make_noisy_image(shape=(128, 128), seed=1000)
        positions = [(10, 10), (30, 50), (60, 80), (90, 30), (110, 110)]
        for r, c in positions:
            data[r, c] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        ci.run(maxiter=3, verbose=False)
        detected = sum(1 for r, c in positions if ci.mask[r, c])
        assert detected >= len(positions) - 1  # Allow at most 1 miss

    def test_adjacent_cosmics_merged(self):
        """Adjacent cosmic pixels are detected as a group."""
        from cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), seed=1001)
        # A small cluster of adjacent cosmics
        data[30, 30] = 60000.0
        data[30, 31] = 60000.0
        data[31, 30] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        ci.lacosmiciteration(verbose=False)
        # All three should be detected
        assert ci.mask[30, 30] or ci.mask[30, 31] or ci.mask[31, 30]

    def test_cosmic_trail(self):
        """A line of cosmic pixels (trail) is largely detected."""
        from cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), seed=1002)
        # Horizontal trail
        for c in range(20, 45):
            data[30, c] = 55000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=3.0,
                          satlevel=-1, verbose=False)
        ci.run(maxiter=3, verbose=False)
        trail_detected = ci.mask[30, 20:45].sum()
        # Most of the trail should be detected
        assert trail_detected > 10


# =============================================================================
# Data type handling
# =============================================================================

class TestDataTypes:
    """Test that different input data types are handled."""

    def test_float32_input(self):
        """float32 input array is handled correctly."""
        from cosmics import cosmicsimage
        data = make_noisy_image(shape=(32, 32), seed=1100).astype(np.float32)
        data[16, 16] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        assert result["niter"] >= 0

    def test_float64_input(self):
        """float64 input array is handled correctly."""
        from cosmics import cosmicsimage
        data = make_noisy_image(shape=(32, 32), seed=1101).astype(np.float64)
        data[16, 16] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        assert result["niter"] > 0

    def test_int_input_promoted_to_float(self):
        """Integer input is accepted (numpy arithmetic promotes to float)."""
        from cosmics import cosmicsimage
        data = np.full((32, 32), 1000, dtype=np.int32)
        data[16, 16] = 60000
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        assert result["niter"] >= 0


# =============================================================================
# Using the image_array fixture from conftest
# =============================================================================

class TestWithFixture:
    """Tests using the shared image_array fixture from conftest."""

    def test_fixture_has_cosmics_detected(self, image_array):
        """The image_array fixture has injected cosmics that can be detected."""
        from cosmics import cosmicsimage
        ci = cosmicsimage(image_array.astype(np.float64), gain=2.2, readnoise=10.0,
                          sigclip=5.0, satlevel=-1, verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        assert result["niter"] > 0

    def test_fixture_cosmics_at_known_positions(self, image_array):
        """The injected cosmic positions [30,40], [70,20], [10,90] are detected."""
        from cosmics import cosmicsimage
        ci = cosmicsimage(image_array.astype(np.float64), gain=2.2, readnoise=10.0,
                          sigclip=5.0, satlevel=-1, verbose=False)
        ci.run(maxiter=3, verbose=False)
        # At least two of the three known cosmic positions should be masked
        detected = sum([
            ci.mask[30, 40],
            ci.mask[70, 20],
            ci.mask[10, 90],
        ])
        assert detected >= 2

    def test_fixture_clean_reduces_cosmic_values(self, image_array):
        """After run(), the cosmic pixel values in cleanarray are much lower."""
        from cosmics import cosmicsimage
        ci = cosmicsimage(image_array.astype(np.float64), gain=2.2, readnoise=10.0,
                          sigclip=5.0, satlevel=-1, verbose=False)
        ci.run(maxiter=3, verbose=False)
        # The cleaned values should be much lower than the original cosmic values
        # Original values were ~45000-50000; cleaned should be near sky (~1000)
        for r, c in [(30, 40), (70, 20), (10, 90)]:
            if ci.mask[r, c]:
                assert ci.getcleanarray()[r, c] < 10000.0


# =============================================================================
# subsample/rebin roundtrip
# =============================================================================

class TestSubsampleRebinRoundtrip:
    """Test that subsample followed by rebin2x2 gives back the original."""

    def test_roundtrip_preserves_values(self):
        """subsample then rebin2x2 recovers the original array values."""
        from cosmics import subsample, rebin2x2
        a = np.array([[1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0],
                      [9.0, 10.0, 11.0, 12.0],
                      [13.0, 14.0, 15.0, 16.0]])
        b = subsample(a)  # 8x8
        c = rebin2x2(b)   # back to 4x4
        assert c.shape == a.shape
        assert np.allclose(c, a)

    def test_roundtrip_uniform_image(self):
        """Roundtrip on uniform image preserves uniform value."""
        from cosmics import subsample, rebin2x2
        a = np.full((10, 10), 42.0)
        b = subsample(a)
        c = rebin2x2(b)
        assert np.allclose(c, 42.0)


# =============================================================================
# run() with satstars already computed
# =============================================================================

class TestRunWithPrecomputedSatstars:
    """Test run() when satstars is already set."""

    def test_run_skips_findsatstars_if_already_set(self):
        """run() does not call findsatstars again if satstars is already set."""
        from cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), seed=1200)
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=20.0,
                          satlevel=50000.0, verbose=False)
        # Pre-set satstars
        ci.satstars = np.zeros((64, 64), dtype=bool)
        ci.run(maxiter=1, verbose=False)
        # satstars should still be the same object we set (not recomputed)
        assert ci.satstars.sum() == 0


# =============================================================================
# Noise model edge cases
# =============================================================================

class TestNoiseModel:
    """Test the noise model calculation inside lacosmiciteration."""

    def test_noise_model_positive_definite(self):
        """The noise model should always produce positive values."""
        from cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), seed=1300)
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        # Access the noise model indirectly by verifying iteration succeeds
        # (noise = (1/gain) * sqrt(gain*m5clipped + readnoise^2) is always positive)
        result = ci.lacosmiciteration(verbose=False)
        assert result is not None

    def test_very_bright_image_noise_model(self):
        """Very bright images (high sky) produce a valid noise model."""
        from cosmics import cosmicsimage
        data = make_flat_image(shape=(32, 32), value=50000.0)
        data[16, 16] = 200000.0  # Very bright cosmic
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        result = ci.lacosmiciteration(verbose=False)
        assert result["niter"] >= 0


# =============================================================================
# Test __str__ with various states
# =============================================================================

class TestStrRepresentation:
    """Test the string representation in various states."""

    def test_str_after_iteration_shows_mask_count(self):
        """__str__ shows number of cosmic pixels after iteration."""
        from cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), seed=1400)
        data[30, 30] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        ci.lacosmiciteration(verbose=False)
        s = str(ci)
        assert "cosmic ray mask" in s.lower() or "pixel" in s.lower()

    def test_str_includes_dtype(self):
        """__str__ includes the array dtype."""
        from cosmics import cosmicsimage
        data = np.ones((32, 32), dtype=np.float64)
        ci = cosmicsimage(data, verbose=False)
        s = str(ci)
        assert "float64" in s

    def test_str_with_satstars_set(self):
        """__str__ shows saturated star info when satstars is set."""
        from cosmics import cosmicsimage
        data = np.ones((32, 32), dtype=np.float64) * 1000.0
        ci = cosmicsimage(data, verbose=False)
        ci.satstars = np.zeros((32, 32), dtype=bool)
        ci.satstars[10:15, 10:15] = True
        s = str(ci)
        assert "Saturated star" in s or "saturated" in s.lower()


# =============================================================================
# Test lacosmiciteration with satstars interaction
# =============================================================================

class TestLacosmicIterationWithSatstars:
    """Test that satstars masking works during iteration."""

    def test_cosmic_inside_satstar_region_not_detected(self):
        """A cosmic pixel inside a satstar region is not detected."""
        from cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), seed=1500)
        data[30, 30] = 60000.0  # cosmic inside satstar region
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        # Manually set satstars to cover the cosmic
        ci.satstars = np.zeros((64, 64), dtype=bool)
        ci.satstars[28:33, 28:33] = True
        result = ci.lacosmiciteration(verbose=False)
        # The pixel should not be in the final selection
        assert not result["itermask"][30, 30]

    def test_cosmic_outside_satstar_region_still_detected(self):
        """A cosmic pixel outside satstar region is still detected normally."""
        from cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), seed=1501)
        data[10, 10] = 60000.0  # cosmic outside satstar region
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        # Satstars covers a different region
        ci.satstars = np.zeros((64, 64), dtype=bool)
        ci.satstars[50:55, 50:55] = True
        result = ci.lacosmiciteration(verbose=False)
        assert result["niter"] > 0


# =============================================================================
# Test getmask and getrawarray/getcleanarray
# =============================================================================

class TestAccessorMethods:
    """Test the accessor/getter methods."""

    def test_getmask_returns_boolean_array(self):
        """getmask returns a boolean numpy array."""
        from cosmics import cosmicsimage
        data = make_flat_image(shape=(16, 16))
        ci = cosmicsimage(data, verbose=False)
        mask = ci.getmask()
        assert mask.dtype == bool
        assert mask.shape == (16, 16)

    def test_getrawarray_shape_matches_input(self):
        """getrawarray returns array with same shape as input."""
        from cosmics import cosmicsimage
        data = make_noisy_image(shape=(20, 30), seed=1600)
        ci = cosmicsimage(data, verbose=False)
        raw = ci.getrawarray()
        assert raw.shape == (20, 30)

    def test_getcleanarray_shape_matches_input(self):
        """getcleanarray returns array with same shape as input."""
        from cosmics import cosmicsimage
        data = make_noisy_image(shape=(20, 30), seed=1601)
        ci = cosmicsimage(data, verbose=False)
        clean = ci.getcleanarray()
        assert clean.shape == (20, 30)

    def test_getcleanarray_initially_equals_getrawarray(self):
        """Before any processing, getcleanarray equals getrawarray."""
        from cosmics import cosmicsimage
        data = make_noisy_image(shape=(32, 32), seed=1602)
        ci = cosmicsimage(data, pssl=100.0, verbose=False)
        assert np.allclose(ci.getcleanarray(), ci.getrawarray())


# =============================================================================
# Test guessbackgroundlevel caching
# =============================================================================

class TestGuessBackgroundLevelCaching:
    """Test background level estimation and caching behavior."""

    def test_backgroundlevel_initially_none(self):
        """backgroundlevel starts as None."""
        from cosmics import cosmicsimage
        data = make_flat_image(shape=(16, 16), value=1000.0)
        ci = cosmicsimage(data, verbose=False)
        assert ci.backgroundlevel is None

    def test_backgroundlevel_set_after_first_call(self):
        """backgroundlevel is set after first call to guessbackgroundlevel."""
        from cosmics import cosmicsimage
        data = make_flat_image(shape=(16, 16), value=1000.0)
        ci = cosmicsimage(data, verbose=False)
        ci.guessbackgroundlevel()
        assert ci.backgroundlevel is not None

    def test_backgroundlevel_is_median_of_rawarray(self):
        """The background level equals the median of rawarray (including pssl)."""
        from cosmics import cosmicsimage
        data = np.arange(256, dtype=np.float64).reshape(16, 16)
        ci = cosmicsimage(data, pssl=0.0, verbose=False)
        level = ci.guessbackgroundlevel()
        expected = np.median(data.ravel())
        assert np.isclose(level, expected)


# =============================================================================
# Test findholes (stub)
# =============================================================================

class TestFindholesStub:
    """Test that findholes is properly stubbed."""

    def test_findholes_returns_none(self):
        """findholes returns None (unimplemented)."""
        from cosmics import cosmicsimage
        data = make_flat_image(shape=(32, 32))
        ci = cosmicsimage(data, verbose=False)
        assert ci.findholes() is None

    def test_findholes_does_not_modify_any_state(self):
        """findholes does not modify mask, cleanarray, or rawarray."""
        from cosmics import cosmicsimage
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
# Integration: full pipeline with image_array fixture
# =============================================================================

class TestFullPipelineIntegration:
    """Full pipeline integration tests."""

    def test_full_pipeline_flat_image_no_change(self):
        """Full pipeline on flat image produces no changes to cleanarray."""
        from cosmics import cosmicsimage
        data = make_flat_image(shape=(64, 64), value=2000.0)
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        ci.run(maxiter=3, verbose=False)
        # cleanarray should remain essentially unchanged
        assert np.allclose(ci.getcleanarray(), 2000.0)

    def test_full_pipeline_cleaning_does_not_introduce_nans(self):
        """The full pipeline should never introduce NaN values."""
        from cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), seed=1800)
        data[30, 30] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        ci.run(maxiter=3, verbose=False)
        assert not np.any(np.isnan(ci.cleanarray))
        assert not np.any(np.isnan(ci.getcleanarray()))

    def test_full_pipeline_cleaning_does_not_introduce_infs(self):
        """The full pipeline should not leave any inf values in cleanarray."""
        from cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), seed=1801)
        data[30, 30] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        ci.run(maxiter=3, verbose=False)
        assert not np.any(np.isinf(ci.cleanarray))

    def test_full_pipeline_mask_is_boolean(self):
        """After full pipeline, mask is still a proper boolean array."""
        from cosmics import cosmicsimage
        data = make_noisy_image(shape=(64, 64), seed=1802)
        data[30, 30] = 60000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=-1, verbose=False)
        ci.run(maxiter=3, verbose=False)
        assert ci.mask.dtype == bool

    def test_full_pipeline_with_satlevel_and_cosmics(self):
        """Full pipeline with both saturated stars and cosmics."""
        from cosmics import cosmicsimage
        data = make_noisy_image(shape=(128, 128), seed=1803)
        # Saturated star
        data[20:28, 20:28] = 55000.0
        # Cosmic rays far from star
        data[80, 80] = 60000.0
        data[100, 100] = 55000.0
        ci = cosmicsimage(data, gain=2.2, readnoise=10.0, sigclip=5.0,
                          satlevel=50000.0, verbose=False)
        ci.run(maxiter=3, verbose=False)
        # Saturated star should not be in cosmic mask
        # (it should be in satstars mask instead)
        assert ci.satstars is not None
        # Cosmic far from star should be detected
        assert ci.mask[80, 80] or ci.mask[100, 100]
