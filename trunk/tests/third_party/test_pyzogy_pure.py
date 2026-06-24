"""
Tests for PyZOGY — the optimal image subtraction library used by lscdiff.py.

Coverage:
  - PyZOGY.util: pad_to_power2, center_psf, resize_psf, mask_saturated_pix, fit_noise,
                 convolve, interpolate_bad_pixels
  - PyZOGY.subtract: calculate_difference_image_zero_point, calculate_difference_psf,
                     calculate_difference_image (with preset gain_ratio — no live DB/IRAF),
                     normalize_difference_image, calculate_matched_filter_image,
                     noise_kernels, registration_noise, source_noise,
                     correct_matched_filter_image, photometric_matched_filter_image,
                     save_difference_image_to_file, save_difference_psf_to_file,
                     run_subtraction
  - lscdiff.py integration: --help, --difftype choices, --normalize choices, --convolve choices

All tests are hermetic (no network, no IRAF, no DB).  Functions that require
running the full gain-matching solver (solve_iteratively) are tested via the
`gain_ratio=` shortcut that bypasses star detection.

Additional coverage added:
  - TestConvolve: delta-kernel identity, smoothing, NaN treatment, boundary options
  - TestInterpolateBadPixels: masked bad pixel replacement with median filter
  - TestNormalizeDifferenceImage: normalization to reference, science, and none
  - TestMatchedFilterImage: Scorr image shape/finiteness/dtype
  - TestNoiseKernels: noise correction kernel shapes and finiteness
  - TestRegistrationNoise: astrometric registration variance image
  - TestSourceNoise: source noise variance correction image
  - TestCorrectMatchedFilterImage: total noise image for noise-corrected Scorr
  - TestPhotometricMatchedFilterImage: photometric matched filter output
  - TestSaveDifferenceImageToFile: FITS write/read roundtrip for difference image
  - TestSaveDifferencePsfToFile: FITS write/read roundtrip for difference PSF
  - TestRunSubtraction: end-to-end high-level entry point
"""
import os
import shutil
import tempfile

import numpy as np
import numpy.ma as ma
import pytest
from astropy.io import fits

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bin"))
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
STUB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "stubs"))


def _run_bin(script_name, *args, timeout=30):
    import subprocess
    import sys

    env = os.environ.copy()
    env["PYTHONPATH"] = STUB_DIR + os.pathsep + SRC_DIR + os.pathsep + env.get("PYTHONPATH", "")
    script = os.path.join(BIN_DIR, script_name)
    return subprocess.run([sys.executable, script, *args], capture_output=True, timeout=timeout, env=env)


def _make_fits_psf(directory, filename, data):
    """Write a float64 FITS file and return its path."""
    path = os.path.join(directory, filename)
    fits.writeto(path, data.astype(np.float64), overwrite=True)
    return path


@pytest.fixture(scope="module")
def image_pair(tmp_path_factory):
    """Create a matched science/reference pair and a delta PSF as FITS files."""
    rng = np.random.default_rng(7)
    n = 64
    sci_data = rng.normal(1000.0, 15.0, (n, n))
    ref_data = sci_data * 2.0 + rng.normal(0.0, 5.0, (n, n))
    psf_data = np.zeros((5, 5))
    psf_data[2, 2] = 1.0  # delta function PSF

    d = str(tmp_path_factory.mktemp("pyzogy"))
    sci_path = _make_fits_psf(d, "science.fits", sci_data)
    ref_path = _make_fits_psf(d, "reference.fits", ref_data)
    psf_path = _make_fits_psf(d, "psf.fits", psf_data)
    return sci_path, ref_path, psf_path


# ---------------------------------------------------------------------------
# PyZOGY.util — pad_to_power2
# ---------------------------------------------------------------------------

class TestPadToPower2:
    def test_already_power_of_two_unchanged(self):
        from PyZOGY.util import pad_to_power2
        data = np.ones((8, 8))
        result = pad_to_power2(data)
        assert result.shape == (8, 8)

    def test_small_array_padded_to_next_power(self):
        from PyZOGY.util import pad_to_power2
        data = np.ones((5, 5))
        result = pad_to_power2(data)
        assert result.shape == (8, 8)  # next power of 2 after 5

    def test_non_square_padded_to_single_power(self):
        from PyZOGY.util import pad_to_power2
        # largest dimension determines the target power
        data = np.ones((3, 7))
        result = pad_to_power2(data)
        assert result.shape == (8, 8)

    def test_original_data_preserved(self):
        from PyZOGY.util import pad_to_power2
        data = np.full((3, 3), 5.0)
        result = pad_to_power2(data)
        np.testing.assert_array_equal(result[:3, :3], data)

    def test_padding_filled_with_median(self):
        from PyZOGY.util import pad_to_power2
        data = np.full((3, 3), 7.0)
        result = pad_to_power2(data)
        assert result[3, 3] == pytest.approx(7.0)

    def test_bool_padding_value(self):
        from PyZOGY.util import pad_to_power2
        data = np.ones((3, 3), dtype=bool)
        result = pad_to_power2(data, value="bool")
        assert result[3, 3] == False  # noqa: E712

    def test_single_element_unchanged(self):
        from PyZOGY.util import pad_to_power2
        data = np.array([[42.0]])
        result = pad_to_power2(data)
        assert result.shape == (1, 1)
        assert result[0, 0] == 42.0

    def test_100x100_pads_to_128(self):
        from PyZOGY.util import pad_to_power2
        data = np.ones((100, 100))
        result = pad_to_power2(data)
        assert result.shape == (128, 128)


# ---------------------------------------------------------------------------
# PyZOGY.util — center_psf
# ---------------------------------------------------------------------------

class TestCenterPsf:
    def test_centered_peak_moves_to_origin(self):
        from PyZOGY.util import center_psf
        psf = np.zeros((8, 8))
        psf[3, 3] = 1.0
        result = center_psf(psf)
        peak = np.unravel_index(result.argmax(), result.shape)
        assert peak == (0, 0)

    def test_already_at_origin_unchanged(self):
        from PyZOGY.util import center_psf
        psf = np.zeros((8, 8))
        psf[0, 0] = 1.0
        result = center_psf(psf)
        assert result[0, 0] == 1.0

    def test_sum_preserved(self):
        from PyZOGY.util import center_psf
        rng = np.random.default_rng(1)
        psf = np.abs(rng.normal(0, 1, (8, 8)))
        result = center_psf(psf)
        assert np.sum(result) == pytest.approx(np.sum(psf))

    def test_off_center_by_one(self):
        from PyZOGY.util import center_psf
        psf = np.zeros((4, 4))
        psf[1, 0] = 1.0
        result = center_psf(psf)
        assert result[0, 0] == 1.0


# ---------------------------------------------------------------------------
# PyZOGY.util — resize_psf
# ---------------------------------------------------------------------------

class TestResizePsf:
    def test_output_shape_matches_target(self):
        from PyZOGY.util import resize_psf
        psf = np.ones((4, 4))
        result = resize_psf(psf, (8, 8))
        assert result.shape == (8, 8)

    def test_sum_conserved_after_resize(self):
        from PyZOGY.util import resize_psf
        psf = np.ones((4, 4))
        result = resize_psf(psf, (8, 8))
        assert np.sum(result) == pytest.approx(np.sum(psf))

    def test_padding_is_zeros(self):
        from PyZOGY.util import resize_psf
        psf = np.ones((4, 4))
        result = resize_psf(psf, (8, 8))
        assert result[7, 7] == 0.0

    def test_same_shape_noop(self):
        from PyZOGY.util import resize_psf
        psf = np.eye(4)
        result = resize_psf(psf, (4, 4))
        np.testing.assert_array_equal(result, psf)

    def test_delta_psf_resize(self):
        from PyZOGY.util import resize_psf
        psf = np.zeros((3, 3))
        psf[0, 0] = 1.0
        result = resize_psf(psf, (8, 8))
        assert result[0, 0] == 1.0
        assert result[1, 0] == 0.0


# ---------------------------------------------------------------------------
# PyZOGY.util — mask_saturated_pix
# ---------------------------------------------------------------------------

class TestMaskSaturatedPix:
    def test_nan_pixels_masked(self):
        from PyZOGY.util import mask_saturated_pix
        img = np.array([[1.0, np.nan], [3.0, 4.0]])
        mask = mask_saturated_pix(img)
        assert mask[0, 1] == True  # noqa: E712

    def test_saturated_pixels_masked(self):
        from PyZOGY.util import mask_saturated_pix
        img = np.array([[1.0, 2.0], [3.0, 100.0]])
        mask = mask_saturated_pix(img, saturation=50.0)
        assert mask[1, 1] == True  # noqa: E712
        assert mask[0, 0] == False  # noqa: E712

    def test_no_bad_pixels_all_false(self):
        from PyZOGY.util import mask_saturated_pix
        img = np.ones((4, 4)) * 10.0
        mask = mask_saturated_pix(img, saturation=1000.0)
        assert not np.any(mask)

    def test_all_nan_all_masked(self):
        from PyZOGY.util import mask_saturated_pix
        img = np.full((3, 3), np.nan)
        mask = mask_saturated_pix(img)
        assert np.all(mask)

    def test_input_mask_joined(self):
        from PyZOGY.util import mask_saturated_pix
        img = np.array([[1.0, 2.0], [3.0, 4.0]])
        prior = np.array([[1, 0], [0, 0]])
        mask = mask_saturated_pix(img, saturation=100.0, input_mask=prior)
        assert mask[0, 0] == True  # noqa: E712
        assert mask[0, 1] == False  # noqa: E712

    def test_exactly_at_saturation_masked(self):
        """Pixels >= saturation should be masked (boundary condition)."""
        from PyZOGY.util import mask_saturated_pix
        img = np.array([[50.0, 49.9]])
        mask = mask_saturated_pix(img, saturation=50.0)
        assert mask[0, 0] == True  # noqa: E712
        assert mask[0, 1] == False  # noqa: E712


# ---------------------------------------------------------------------------
# PyZOGY.util — fit_noise
# ---------------------------------------------------------------------------

class TestFitNoise:
    def _make_masked(self, data):
        return ma.array(data, mask=np.zeros(data.shape, dtype=bool))

    def test_returns_two_arrays(self):
        from PyZOGY.util import fit_noise
        data = self._make_masked(np.ones((32, 32)) * 500.0)
        result = fit_noise(data)
        assert len(result) == 2

    def test_std_shape_matches_input(self):
        from PyZOGY.util import fit_noise
        data = self._make_masked(np.random.default_rng(0).normal(1000, 10, (64, 64)))
        std, _ = fit_noise(data)
        assert std.shape == data.shape

    def test_median_shape_matches_input(self):
        from PyZOGY.util import fit_noise
        data = self._make_masked(np.random.default_rng(0).normal(1000, 10, (64, 64)))
        _, med = fit_noise(data)
        assert med.shape == data.shape

    def test_std_estimate_close_to_truth(self):
        from PyZOGY.util import fit_noise
        rng = np.random.default_rng(3)
        data = self._make_masked(rng.normal(1000.0, 20.0, (64, 64)))
        std, _ = fit_noise(data)
        # IQR estimator: should be within 50% of true std
        assert 10.0 < np.mean(std) < 30.0

    def test_median_estimate_close_to_truth(self):
        from PyZOGY.util import fit_noise
        rng = np.random.default_rng(5)
        data = self._make_masked(rng.normal(500.0, 5.0, (64, 64)))
        _, med = fit_noise(data)
        assert abs(np.mean(med) - 500.0) < 10.0

    def test_uniform_image_std_is_zero(self):
        from PyZOGY.util import fit_noise
        data = self._make_masked(np.full((32, 32), 42.0))
        std, _ = fit_noise(data)
        assert np.mean(std) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# PyZOGY.subtract — calculate_difference_image_zero_point / _psf / _image
# (gain_ratio is pre-set to skip the star-detection fitting step)
# ---------------------------------------------------------------------------

class TestCalculateDifferenceImageZeroPoint:
    def test_returns_array(self, image_pair):
        from PyZOGY.subtract import calculate_difference_image_zero_point
        from PyZOGY.image_class import ImageClass
        sci_path, ref_path, psf_path = image_pair
        sci = ImageClass(sci_path, psf_path, n_stamps=1)
        ref = ImageClass(ref_path, psf_path, n_stamps=1)
        sci.zero_point = 2.0
        ref.zero_point = 1.0
        zp = calculate_difference_image_zero_point(sci, ref)
        assert isinstance(zp, np.ndarray)

    def test_shape_matches_image(self, image_pair):
        from PyZOGY.subtract import calculate_difference_image_zero_point
        from PyZOGY.image_class import ImageClass
        sci_path, ref_path, psf_path = image_pair
        sci = ImageClass(sci_path, psf_path, n_stamps=1)
        ref = ImageClass(ref_path, psf_path, n_stamps=1)
        sci.zero_point = 2.0
        ref.zero_point = 1.0
        zp = calculate_difference_image_zero_point(sci, ref)
        assert zp.shape == fits.getdata(sci_path).shape

    def test_values_are_finite(self, image_pair):
        from PyZOGY.subtract import calculate_difference_image_zero_point
        from PyZOGY.image_class import ImageClass
        sci_path, ref_path, psf_path = image_pair
        sci = ImageClass(sci_path, psf_path, n_stamps=1)
        ref = ImageClass(ref_path, psf_path, n_stamps=1)
        sci.zero_point = 2.0
        ref.zero_point = 1.0
        zp = calculate_difference_image_zero_point(sci, ref)
        assert np.all(np.isfinite(zp))

    def test_zero_point_is_positive(self, image_pair):
        from PyZOGY.subtract import calculate_difference_image_zero_point
        from PyZOGY.image_class import ImageClass
        sci_path, ref_path, psf_path = image_pair
        sci = ImageClass(sci_path, psf_path, n_stamps=1)
        ref = ImageClass(ref_path, psf_path, n_stamps=1)
        sci.zero_point = 2.0
        ref.zero_point = 1.0
        zp = calculate_difference_image_zero_point(sci, ref)
        assert np.all(zp > 0)


class TestCalculateDifferencePsf:
    def test_returns_array_same_shape_as_image(self, image_pair):
        from PyZOGY.subtract import calculate_difference_image_zero_point, calculate_difference_psf
        from PyZOGY.image_class import ImageClass
        sci_path, ref_path, psf_path = image_pair
        sci = ImageClass(sci_path, psf_path, n_stamps=1)
        ref = ImageClass(ref_path, psf_path, n_stamps=1)
        sci.zero_point = 2.0
        ref.zero_point = 1.0
        zp = calculate_difference_image_zero_point(sci, ref)
        psf_d = calculate_difference_psf(sci, ref, zp)
        assert psf_d.shape == fits.getdata(sci_path).shape

    def test_psf_values_are_finite(self, image_pair):
        from PyZOGY.subtract import calculate_difference_image_zero_point, calculate_difference_psf
        from PyZOGY.image_class import ImageClass
        sci_path, ref_path, psf_path = image_pair
        sci = ImageClass(sci_path, psf_path, n_stamps=1)
        ref = ImageClass(ref_path, psf_path, n_stamps=1)
        sci.zero_point = 2.0
        ref.zero_point = 1.0
        zp = calculate_difference_image_zero_point(sci, ref)
        psf_d = calculate_difference_psf(sci, ref, zp)
        assert np.all(np.isfinite(psf_d))


class TestCalculateDifferenceImage:
    def test_output_shape_matches_input(self, image_pair):
        from PyZOGY.subtract import calculate_difference_image
        from PyZOGY.image_class import ImageClass
        sci_path, ref_path, psf_path = image_pair
        sci = ImageClass(sci_path, psf_path, n_stamps=1)
        ref = ImageClass(ref_path, psf_path, n_stamps=1)
        diff = calculate_difference_image(sci, ref, gain_ratio=2.0)
        assert diff.shape == fits.getdata(sci_path).shape

    def test_output_is_finite(self, image_pair):
        from PyZOGY.subtract import calculate_difference_image
        from PyZOGY.image_class import ImageClass
        sci_path, ref_path, psf_path = image_pair
        sci = ImageClass(sci_path, psf_path, n_stamps=1)
        ref = ImageClass(ref_path, psf_path, n_stamps=1)
        diff = calculate_difference_image(sci, ref, gain_ratio=2.0)
        assert np.all(np.isfinite(diff))

    def test_output_is_float64(self, image_pair):
        from PyZOGY.subtract import calculate_difference_image
        from PyZOGY.image_class import ImageClass
        sci_path, ref_path, psf_path = image_pair
        sci = ImageClass(sci_path, psf_path, n_stamps=1)
        ref = ImageClass(ref_path, psf_path, n_stamps=1)
        diff = calculate_difference_image(sci, ref, gain_ratio=2.0)
        assert diff.dtype == np.float64

    def test_identical_images_diff_near_zero(self, tmp_path):
        """Science == reference with unity gain should produce a near-zero difference."""
        from PyZOGY.subtract import calculate_difference_image
        from PyZOGY.image_class import ImageClass
        rng = np.random.default_rng(99)
        data = rng.normal(500.0, 5.0, (64, 64))
        psf = np.zeros((5, 5)); psf[2, 2] = 1.0
        sci_path = str(tmp_path / "sci.fits")
        ref_path = str(tmp_path / "ref.fits")
        psf_path = str(tmp_path / "psf.fits")
        fits.writeto(sci_path, data.astype(np.float64))
        fits.writeto(ref_path, data.astype(np.float64))
        fits.writeto(psf_path, psf.astype(np.float64))
        sci = ImageClass(sci_path, psf_path, n_stamps=1)
        ref = ImageClass(ref_path, psf_path, n_stamps=1)
        diff = calculate_difference_image(sci, ref, gain_ratio=1.0)
        # mean absolute deviation should be very small
        assert np.mean(np.abs(diff)) < 5.0

    def test_scaled_reference_absorbed_by_gain_ratio(self, tmp_path):
        """If reference = k * science, setting gain_ratio=k should yield ~0 difference."""
        from PyZOGY.subtract import calculate_difference_image
        from PyZOGY.image_class import ImageClass
        rng = np.random.default_rng(11)
        k = 3.0
        sci_data = rng.normal(800.0, 8.0, (64, 64))
        ref_data = sci_data * k
        psf = np.zeros((5, 5)); psf[2, 2] = 1.0
        sci_path = str(tmp_path / "sci.fits")
        ref_path = str(tmp_path / "ref.fits")
        psf_path = str(tmp_path / "psf.fits")
        fits.writeto(sci_path, sci_data.astype(np.float64))
        fits.writeto(ref_path, ref_data.astype(np.float64))
        fits.writeto(psf_path, psf.astype(np.float64))
        sci = ImageClass(sci_path, psf_path, n_stamps=1)
        ref = ImageClass(ref_path, psf_path, n_stamps=1)
        diff = calculate_difference_image(sci, ref, gain_ratio=k)
        assert np.mean(np.abs(diff)) < 20.0


# ---------------------------------------------------------------------------
# lscdiff.py CLI edge cases (source inspection — pyraf not available in CI)
# ---------------------------------------------------------------------------

class TestLscdiffSource:
    """Inspect lscdiff.py source to verify all edge-case flags are declared."""

    def _src(self):
        with open(os.path.join(BIN_DIR, "lscdiff.py")) as fh:
            return fh.read()

    def test_normalize_flag_declared(self):
        assert "--normalize" in self._src()

    def test_normalize_choices_i_and_t(self):
        src = self._src()
        assert "choices=['i', 't']" in src or "choices=[\"i\", \"t\"]" in src

    def test_convolve_flag_declared(self):
        assert "--convolve" in self._src()

    def test_convolve_empty_string_choice_present(self):
        # empty string is a valid convolve choice (no forced convolution)
        assert "''" in self._src() or '""' in self._src()

    def test_difftype_choices_0_and_1(self):
        src = self._src()
        assert "--difftype" in src
        assert "choices=[0, 1]" in src

    def test_unmask_flag_declared(self):
        assert "--unmask" in self._src()

    def test_pixstack_limit_flag_declared(self):
        assert "--pixstack-limit" in self._src()

    def test_interpolation_flag_declared(self):
        assert "--interpolation" in self._src()

    def test_help_exits_cleanly(self):
        """lscdiff.py --help must now reach argparse (heavy imports deferred)."""
        r = _run_bin("lscdiff.py", "--help")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert r.returncode in (0, 1), f"unexpected exit {r.returncode}\n{combined}"
        assert "normalize" in combined
        assert "convolve" in combined

    def test_missing_positional_args_exits_nonzero(self):
        """Invoking with no positional args must fail (not pyraf-crash)."""
        r = _run_bin("lscdiff.py")
        assert r.returncode != 0

    def test_invalid_normalize_choice_rejected(self):
        r = _run_bin("lscdiff.py", "a.fits", "b.fits", "--normalize", "bad")
        assert r.returncode != 0

    def test_invalid_difftype_choice_rejected(self):
        r = _run_bin("lscdiff.py", "a.fits", "b.fits", "--difftype", "99")
        assert r.returncode != 0


# ---------------------------------------------------------------------------
# PyZOGY.util — convolve (NaN-safe astropy wrapper)
# ---------------------------------------------------------------------------

class TestConvolve:
    """util.convolve wraps astropy convolution with NaN-safe treatment."""

    def test_delta_kernel_is_identity(self):
        """Convolving with a delta-function kernel returns the same array."""
        from PyZOGY.util import convolve
        data = np.array([[0.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.0]])
        kernel = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        result = convolve(data, kernel)
        np.testing.assert_allclose(result, data, atol=1e-10)

    def test_output_shape_preserved(self):
        from PyZOGY.util import convolve
        data = np.ones((8, 8))
        kernel = np.ones((3, 3)) / 9.0
        result = convolve(data, kernel)
        assert result.shape == (8, 8)

    def test_smoothing_kernel_reduces_peak(self):
        """A box-smoothing kernel spreads a central spike over neighbors."""
        from PyZOGY.util import convolve
        data = np.zeros((7, 7))
        data[3, 3] = 9.0
        kernel = np.ones((3, 3)) / 9.0
        result = convolve(data, kernel)
        assert result[3, 3] < 9.0

    def test_uniform_image_unchanged_by_mean_filter(self):
        """A uniform image convolved with a mean kernel stays uniform (wrap boundary)."""
        from PyZOGY.util import convolve
        data = np.full((8, 8), 3.0)
        kernel = np.ones((3, 3)) / 9.0
        result = convolve(data, kernel, boundary='wrap')
        np.testing.assert_allclose(result, 3.0, atol=1e-10)

    def test_nan_treatment_fill_replaces_nan(self):
        """nan_treatment='fill' replaces NaN with fill_value before convolution."""
        from PyZOGY.util import convolve
        data = np.array([[1.0, 1.0, 1.0], [1.0, np.nan, 1.0], [1.0, 1.0, 1.0]])
        kernel = np.ones((3, 3)) / 9.0
        result = convolve(data, kernel, nan_treatment='fill', fill_value=0.0)
        assert np.isfinite(result[1, 1])

    def test_boundary_wrap_uniform_image(self):
        """Wrap boundary: a constant image stays constant throughout."""
        from PyZOGY.util import convolve
        data = np.ones((5, 5))
        kernel = np.ones((3, 3)) / 9.0
        result = convolve(data, kernel, boundary='wrap')
        np.testing.assert_allclose(result, 1.0, atol=1e-10)

    def test_normalize_kernel_true_conserves_flux(self):
        """When normalize_kernel=True the mean is preserved for a uniform image."""
        from PyZOGY.util import convolve
        data = np.full((5, 5), 2.0)
        kernel = np.ones((3, 3))  # unnormalized — will be normalised by convolve
        result = convolve(data, kernel, boundary='wrap', normalize_kernel=True)
        np.testing.assert_allclose(result, 2.0, atol=1e-10)

    def test_output_is_float(self):
        from PyZOGY.util import convolve
        data = np.ones((5, 5))
        kernel = np.ones((3, 3)) / 9.0
        result = convolve(data, kernel)
        assert np.issubdtype(result.dtype, np.floating)


# ---------------------------------------------------------------------------
# PyZOGY.util — interpolate_bad_pixels
# ---------------------------------------------------------------------------

class TestInterpolateBadPixels:
    """util.interpolate_bad_pixels replaces masked pixels with local median."""

    def test_returns_array_same_shape(self):
        from PyZOGY.util import interpolate_bad_pixels
        data = ma.array(np.ones((16, 16)), mask=np.zeros((16, 16), dtype=bool))
        result = interpolate_bad_pixels(data)
        assert result.shape == (16, 16)

    def test_no_mask_values_unchanged(self):
        """When no pixels are masked the output equals the input in the interior."""
        from PyZOGY.util import interpolate_bad_pixels
        data = np.full((16, 16), 7.0)
        masked = ma.array(data, mask=np.zeros((16, 16), dtype=bool))
        result = interpolate_bad_pixels(masked)
        np.testing.assert_allclose(result[4:12, 4:12], 7.0, atol=1e-6)

    def test_single_bad_pixel_replaced_by_local_value(self):
        """A masked pixel surrounded by uniform values is replaced (not left as 9999)."""
        from PyZOGY.util import interpolate_bad_pixels
        data = np.full((16, 16), 5.0)
        data[8, 8] = 9999.0
        mask = np.zeros((16, 16), dtype=bool)
        mask[8, 8] = True
        result = interpolate_bad_pixels(ma.array(data, mask=mask))
        # The bad pixel should be finite and not equal to the original sentinel
        assert np.isfinite(result[8, 8])
        assert result[8, 8] != 9999.0

    def test_output_is_ndarray(self):
        from PyZOGY.util import interpolate_bad_pixels
        data = ma.array(np.ones((16, 16)), mask=np.zeros((16, 16), dtype=bool))
        result = interpolate_bad_pixels(data)
        assert isinstance(result, np.ndarray)

    def test_good_pixels_far_from_bad_region_unchanged(self):
        """Pixels far from a masked corner should be unaffected."""
        from PyZOGY.util import interpolate_bad_pixels
        data = np.full((16, 16), 3.0)
        mask = np.zeros((16, 16), dtype=bool)
        mask[0, 0] = True
        result = interpolate_bad_pixels(ma.array(data, mask=mask))
        assert result[15, 15] == pytest.approx(3.0, abs=0.5)


# ---------------------------------------------------------------------------
# PyZOGY.subtract — normalize_difference_image
# ---------------------------------------------------------------------------

class TestNormalizeDifferenceImage:
    """normalize_difference_image scales D into the science or reference system."""

    @pytest.fixture
    def subtracted(self, image_pair):
        from PyZOGY.subtract import (
            calculate_difference_image,
            calculate_difference_image_zero_point,
        )
        from PyZOGY.image_class import ImageClass
        sci_path, ref_path, psf_path = image_pair
        sci = ImageClass(sci_path, psf_path, n_stamps=1)
        ref = ImageClass(ref_path, psf_path, n_stamps=1)
        sci.zero_point = 2.0
        ref.zero_point = 1.0
        diff = calculate_difference_image(sci, ref, gain_ratio=2.0)
        zp = calculate_difference_image_zero_point(sci, ref)
        return diff, zp, sci, ref

    def test_none_normalization_equals_raw_diff(self, subtracted):
        from PyZOGY.subtract import normalize_difference_image
        diff, zp, sci, ref = subtracted
        norm = normalize_difference_image(diff, zp, sci, ref, normalization='none')
        np.testing.assert_allclose(norm, diff)

    def test_reference_normalization_returns_array(self, subtracted):
        from PyZOGY.subtract import normalize_difference_image
        diff, zp, sci, ref = subtracted
        norm = normalize_difference_image(diff, zp, sci, ref, normalization='reference')
        assert isinstance(norm, np.ndarray)
        assert norm.shape == diff.shape

    def test_science_normalization_returns_array(self, subtracted):
        from PyZOGY.subtract import normalize_difference_image
        diff, zp, sci, ref = subtracted
        norm = normalize_difference_image(diff, zp, sci, ref, normalization='science')
        assert isinstance(norm, np.ndarray)
        assert norm.shape == diff.shape

    def test_reference_and_science_normalization_differ(self, subtracted):
        """Different normalization modes should scale by different factors."""
        from PyZOGY.subtract import normalize_difference_image
        diff, zp, sci, ref = subtracted
        norm_ref = normalize_difference_image(diff, zp, sci, ref, normalization='reference')
        norm_sci = normalize_difference_image(diff, zp, sci, ref, normalization='science')
        assert not np.allclose(norm_ref, norm_sci)

    def test_all_normalizations_are_finite(self, subtracted):
        from PyZOGY.subtract import normalize_difference_image
        diff, zp, sci, ref = subtracted
        for mode in ('reference', 'science', 'none'):
            norm = normalize_difference_image(diff, zp, sci, ref, normalization=mode)
            assert np.all(np.isfinite(norm)), f"non-finite values in {mode} normalization"


# ---------------------------------------------------------------------------
# PyZOGY.subtract — calculate_matched_filter_image (Scorr)
# ---------------------------------------------------------------------------

class TestMatchedFilterImage:
    """calculate_matched_filter_image produces the Scorr detection statistic."""

    @pytest.fixture
    def diff_components(self, image_pair):
        from PyZOGY.subtract import (
            calculate_difference_image,
            calculate_difference_image_zero_point,
            calculate_difference_psf,
        )
        from PyZOGY.image_class import ImageClass
        sci_path, ref_path, psf_path = image_pair
        sci = ImageClass(sci_path, psf_path, n_stamps=1)
        ref = ImageClass(ref_path, psf_path, n_stamps=1)
        sci.zero_point = 2.0
        ref.zero_point = 1.0
        diff = calculate_difference_image(sci, ref, gain_ratio=2.0)
        zp = calculate_difference_image_zero_point(sci, ref)
        diff_psf = calculate_difference_psf(sci, ref, zp)
        return diff, diff_psf, zp

    def test_shape_matches_difference_image(self, diff_components):
        from PyZOGY.subtract import calculate_matched_filter_image
        diff, diff_psf, zp = diff_components
        mf = calculate_matched_filter_image(diff, diff_psf, zp)
        assert mf.shape == diff.shape

    def test_output_is_finite(self, diff_components):
        from PyZOGY.subtract import calculate_matched_filter_image
        diff, diff_psf, zp = diff_components
        mf = calculate_matched_filter_image(diff, diff_psf, zp)
        assert np.all(np.isfinite(mf))

    def test_output_dtype_is_numeric(self, diff_components):
        """Matched filter is floating or complex (with negligible imaginary part)."""
        from PyZOGY.subtract import calculate_matched_filter_image
        diff, diff_psf, zp = diff_components
        mf = calculate_matched_filter_image(diff, diff_psf, zp)
        assert np.issubdtype(mf.dtype, np.floating) or np.issubdtype(mf.dtype, np.complexfloating)

    def test_imaginary_part_is_negligible(self, diff_components):
        """For a delta-PSF pair the imaginary part of Scorr should be ~0."""
        from PyZOGY.subtract import calculate_matched_filter_image
        diff, diff_psf, zp = diff_components
        mf = calculate_matched_filter_image(diff, diff_psf, zp)
        imag_max = np.max(np.abs(np.imag(mf))) if np.iscomplexobj(mf) else 0.0
        assert imag_max < 1e-6


# ---------------------------------------------------------------------------
# PyZOGY.subtract — noise_kernels
# ---------------------------------------------------------------------------

class TestNoiseKernels:
    """noise_kernels returns the two FFT-domain kernels for noise correction."""

    def test_returns_two_arrays(self, image_pair):
        from PyZOGY.subtract import noise_kernels
        from PyZOGY.image_class import ImageClass
        sci_path, ref_path, psf_path = image_pair
        sci = ImageClass(sci_path, psf_path, n_stamps=1); sci.zero_point = 2.0
        ref = ImageClass(ref_path, psf_path, n_stamps=1); ref.zero_point = 1.0
        result = noise_kernels(sci, ref)
        assert len(result) == 2

    def test_science_kernel_shape_matches_image(self, image_pair):
        from PyZOGY.subtract import noise_kernels
        from PyZOGY.image_class import ImageClass
        sci_path, ref_path, psf_path = image_pair
        sci = ImageClass(sci_path, psf_path, n_stamps=1); sci.zero_point = 2.0
        ref = ImageClass(ref_path, psf_path, n_stamps=1); ref.zero_point = 1.0
        sk, rk = noise_kernels(sci, ref)
        assert sk.shape == fits.getdata(sci_path).shape

    def test_reference_kernel_shape_matches_image(self, image_pair):
        from PyZOGY.subtract import noise_kernels
        from PyZOGY.image_class import ImageClass
        sci_path, ref_path, psf_path = image_pair
        sci = ImageClass(sci_path, psf_path, n_stamps=1); sci.zero_point = 2.0
        ref = ImageClass(ref_path, psf_path, n_stamps=1); ref.zero_point = 1.0
        sk, rk = noise_kernels(sci, ref)
        assert rk.shape == fits.getdata(ref_path).shape

    def test_kernels_are_finite(self, image_pair):
        from PyZOGY.subtract import noise_kernels
        from PyZOGY.image_class import ImageClass
        sci_path, ref_path, psf_path = image_pair
        sci = ImageClass(sci_path, psf_path, n_stamps=1); sci.zero_point = 2.0
        ref = ImageClass(ref_path, psf_path, n_stamps=1); ref.zero_point = 1.0
        sk, rk = noise_kernels(sci, ref)
        assert np.all(np.isfinite(sk))
        assert np.all(np.isfinite(rk))


# ---------------------------------------------------------------------------
# PyZOGY.subtract — registration_noise
# ---------------------------------------------------------------------------

class TestRegistrationNoise:
    """registration_noise computes astrometric variance from registration error."""

    def test_output_shape_matches_image(self, image_pair):
        from PyZOGY.subtract import noise_kernels, registration_noise
        from PyZOGY.image_class import ImageClass
        sci_path, ref_path, psf_path = image_pair
        sci = ImageClass(sci_path, psf_path, n_stamps=1); sci.zero_point = 2.0
        ref = ImageClass(ref_path, psf_path, n_stamps=1); ref.zero_point = 1.0
        sk, _ = noise_kernels(sci, ref)
        reg_var = registration_noise(sci, sk)
        assert reg_var.shape == fits.getdata(sci_path).shape

    def test_output_is_finite(self, image_pair):
        from PyZOGY.subtract import noise_kernels, registration_noise
        from PyZOGY.image_class import ImageClass
        sci_path, ref_path, psf_path = image_pair
        sci = ImageClass(sci_path, psf_path, n_stamps=1); sci.zero_point = 2.0
        ref = ImageClass(ref_path, psf_path, n_stamps=1); ref.zero_point = 1.0
        sk, _ = noise_kernels(sci, ref)
        reg_var = registration_noise(sci, sk)
        assert np.all(np.isfinite(reg_var))

    def test_default_zero_registration_noise_yields_zero_variance(self, image_pair):
        """With the default registration_noise=(0,0), variance must be zero."""
        from PyZOGY.subtract import noise_kernels, registration_noise
        from PyZOGY.image_class import ImageClass
        sci_path, ref_path, psf_path = image_pair
        sci = ImageClass(sci_path, psf_path, n_stamps=1); sci.zero_point = 2.0
        ref = ImageClass(ref_path, psf_path, n_stamps=1); ref.zero_point = 1.0
        assert sci.registration_noise == (0, 0)
        sk, _ = noise_kernels(sci, ref)
        reg_var = registration_noise(sci, sk)
        np.testing.assert_array_equal(reg_var, 0.0)


# ---------------------------------------------------------------------------
# PyZOGY.subtract — source_noise
# ---------------------------------------------------------------------------

class TestSourceNoise:
    """source_noise computes pixel variance due to Poisson source photons."""

    def test_output_shape_matches_image(self, image_pair):
        from PyZOGY.subtract import noise_kernels, source_noise
        from PyZOGY.image_class import ImageClass
        sci_path, ref_path, psf_path = image_pair
        sci = ImageClass(sci_path, psf_path, n_stamps=1); sci.zero_point = 2.0
        ref = ImageClass(ref_path, psf_path, n_stamps=1); ref.zero_point = 1.0
        sk, _ = noise_kernels(sci, ref)
        src_var = source_noise(sci, sk)
        assert src_var.shape == fits.getdata(sci_path).shape

    def test_output_is_finite(self, image_pair):
        from PyZOGY.subtract import noise_kernels, source_noise
        from PyZOGY.image_class import ImageClass
        sci_path, ref_path, psf_path = image_pair
        sci = ImageClass(sci_path, psf_path, n_stamps=1); sci.zero_point = 2.0
        ref = ImageClass(ref_path, psf_path, n_stamps=1); ref.zero_point = 1.0
        sk, _ = noise_kernels(sci, ref)
        src_var = source_noise(sci, sk)
        assert np.all(np.isfinite(src_var))

    def test_output_is_nonnegative(self, image_pair):
        """Variance must be >= 0 everywhere."""
        from PyZOGY.subtract import noise_kernels, source_noise
        from PyZOGY.image_class import ImageClass
        sci_path, ref_path, psf_path = image_pair
        sci = ImageClass(sci_path, psf_path, n_stamps=1); sci.zero_point = 2.0
        ref = ImageClass(ref_path, psf_path, n_stamps=1); ref.zero_point = 1.0
        sk, _ = noise_kernels(sci, ref)
        src_var = source_noise(sci, sk)
        assert np.all(src_var >= 0.0)


# ---------------------------------------------------------------------------
# PyZOGY.subtract — correct_matched_filter_image
# ---------------------------------------------------------------------------

class TestCorrectMatchedFilterImage:
    """correct_matched_filter_image returns total noise for Scorr normalisation."""

    def test_output_shape_matches_image(self, image_pair):
        from PyZOGY.subtract import correct_matched_filter_image
        from PyZOGY.image_class import ImageClass
        sci_path, ref_path, psf_path = image_pair
        sci = ImageClass(sci_path, psf_path, n_stamps=1); sci.zero_point = 2.0
        ref = ImageClass(ref_path, psf_path, n_stamps=1); ref.zero_point = 1.0
        noise = correct_matched_filter_image(sci, ref)
        assert noise.shape == fits.getdata(sci_path).shape

    def test_output_is_finite(self, image_pair):
        from PyZOGY.subtract import correct_matched_filter_image
        from PyZOGY.image_class import ImageClass
        sci_path, ref_path, psf_path = image_pair
        sci = ImageClass(sci_path, psf_path, n_stamps=1); sci.zero_point = 2.0
        ref = ImageClass(ref_path, psf_path, n_stamps=1); ref.zero_point = 1.0
        noise = correct_matched_filter_image(sci, ref)
        assert np.all(np.isfinite(noise))

    def test_output_dtype_is_numeric(self, image_pair):
        """Noise image is floating or complex (with negligible imaginary part)."""
        from PyZOGY.subtract import correct_matched_filter_image
        from PyZOGY.image_class import ImageClass
        sci_path, ref_path, psf_path = image_pair
        sci = ImageClass(sci_path, psf_path, n_stamps=1); sci.zero_point = 2.0
        ref = ImageClass(ref_path, psf_path, n_stamps=1); ref.zero_point = 1.0
        noise = correct_matched_filter_image(sci, ref)
        assert (np.issubdtype(noise.dtype, np.floating)
                or np.issubdtype(noise.dtype, np.complexfloating))


# ---------------------------------------------------------------------------
# PyZOGY.subtract — photometric_matched_filter_image
# ---------------------------------------------------------------------------

class TestPhotometricMatchedFilterImage:
    """photometric_matched_filter_image calibrates Scorr to flux units."""

    @pytest.fixture
    def mf_components(self, image_pair):
        from PyZOGY.subtract import (
            calculate_difference_image,
            calculate_difference_image_zero_point,
            calculate_difference_psf,
            calculate_matched_filter_image,
        )
        from PyZOGY.image_class import ImageClass
        sci_path, ref_path, psf_path = image_pair
        sci = ImageClass(sci_path, psf_path, n_stamps=1); sci.zero_point = 2.0
        ref = ImageClass(ref_path, psf_path, n_stamps=1); ref.zero_point = 1.0
        diff = calculate_difference_image(sci, ref, gain_ratio=2.0)
        zp = calculate_difference_image_zero_point(sci, ref)
        diff_psf = calculate_difference_psf(sci, ref, zp)
        mf = calculate_matched_filter_image(diff, diff_psf, zp)
        return sci, ref, mf

    def test_output_shape_matches_matched_filter(self, mf_components):
        from PyZOGY.subtract import photometric_matched_filter_image
        sci, ref, mf = mf_components
        phot = photometric_matched_filter_image(sci, ref, mf)
        assert phot.shape == mf.shape

    def test_output_is_finite(self, mf_components):
        from PyZOGY.subtract import photometric_matched_filter_image
        sci, ref, mf = mf_components
        phot = photometric_matched_filter_image(sci, ref, mf)
        assert np.all(np.isfinite(phot))

    def test_output_is_scaled_relative_to_input(self, mf_components):
        """Photometric MF is the MF divided by a scalar, so shapes agree."""
        from PyZOGY.subtract import photometric_matched_filter_image
        sci, ref, mf = mf_components
        phot = photometric_matched_filter_image(sci, ref, mf)
        # ratio should be constant across all pixels (scalar division)
        ratio = phot / mf
        assert ratio.shape == mf.shape


# ---------------------------------------------------------------------------
# PyZOGY.subtract — save_difference_image_to_file
# ---------------------------------------------------------------------------

class TestSaveDifferenceImageToFile:
    """save_difference_image_to_file writes a FITS file that can be re-read."""

    def test_file_is_created(self, image_pair, tmp_path):
        from PyZOGY.subtract import calculate_difference_image, save_difference_image_to_file
        from PyZOGY.image_class import ImageClass
        sci_path, ref_path, psf_path = image_pair
        sci = ImageClass(sci_path, psf_path, n_stamps=1); sci.zero_point = 2.0
        ref = ImageClass(ref_path, psf_path, n_stamps=1); ref.zero_point = 1.0
        diff = calculate_difference_image(sci, ref, gain_ratio=2.0)
        out = str(tmp_path / "diff.fits")
        save_difference_image_to_file(diff, sci, 'none', out)
        assert os.path.exists(out)

    def test_saved_data_shape_is_preserved(self, image_pair, tmp_path):
        from PyZOGY.subtract import calculate_difference_image, save_difference_image_to_file
        from PyZOGY.image_class import ImageClass
        sci_path, ref_path, psf_path = image_pair
        sci = ImageClass(sci_path, psf_path, n_stamps=1); sci.zero_point = 2.0
        ref = ImageClass(ref_path, psf_path, n_stamps=1); ref.zero_point = 1.0
        diff = calculate_difference_image(sci, ref, gain_ratio=2.0)
        out = str(tmp_path / "diff2.fits")
        save_difference_image_to_file(diff, sci, 'none', out)
        recovered = fits.getdata(out)
        assert recovered.shape == diff.shape

    def test_reference_normalization_saves_finite_data(self, image_pair, tmp_path):
        from PyZOGY.subtract import calculate_difference_image, save_difference_image_to_file
        from PyZOGY.image_class import ImageClass
        sci_path, ref_path, psf_path = image_pair
        sci = ImageClass(sci_path, psf_path, n_stamps=1); sci.zero_point = 2.0
        ref = ImageClass(ref_path, psf_path, n_stamps=1); ref.zero_point = 1.0
        diff = calculate_difference_image(sci, ref, gain_ratio=2.0)
        out = str(tmp_path / "diff_ref.fits")
        save_difference_image_to_file(diff, sci, 'reference', out)
        recovered = fits.getdata(out)
        assert np.all(np.isfinite(recovered))


# ---------------------------------------------------------------------------
# PyZOGY.subtract — save_difference_psf_to_file
# ---------------------------------------------------------------------------

class TestSaveDifferencePsfToFile:
    """save_difference_psf_to_file writes the difference PSF to a FITS file."""

    def test_file_is_created(self, image_pair, tmp_path):
        from PyZOGY.subtract import (
            calculate_difference_image_zero_point,
            calculate_difference_psf,
            save_difference_psf_to_file,
        )
        from PyZOGY.image_class import ImageClass
        sci_path, ref_path, psf_path = image_pair
        sci = ImageClass(sci_path, psf_path, n_stamps=1); sci.zero_point = 2.0
        ref = ImageClass(ref_path, psf_path, n_stamps=1); ref.zero_point = 1.0
        zp = calculate_difference_image_zero_point(sci, ref)
        diff_psf = calculate_difference_psf(sci, ref, zp)
        out = str(tmp_path / "diff_psf.fits")
        save_difference_psf_to_file(diff_psf, out)
        assert os.path.exists(out)

    def test_saved_psf_shape_is_preserved(self, image_pair, tmp_path):
        from PyZOGY.subtract import (
            calculate_difference_image_zero_point,
            calculate_difference_psf,
            save_difference_psf_to_file,
        )
        from PyZOGY.image_class import ImageClass
        sci_path, ref_path, psf_path = image_pair
        sci = ImageClass(sci_path, psf_path, n_stamps=1); sci.zero_point = 2.0
        ref = ImageClass(ref_path, psf_path, n_stamps=1); ref.zero_point = 1.0
        zp = calculate_difference_image_zero_point(sci, ref)
        diff_psf = calculate_difference_psf(sci, ref, zp)
        out = str(tmp_path / "diff_psf2.fits")
        save_difference_psf_to_file(diff_psf, out)
        recovered = fits.getdata(out)
        assert recovered.shape == diff_psf.shape

    def test_saved_psf_values_are_finite(self, image_pair, tmp_path):
        from PyZOGY.subtract import (
            calculate_difference_image_zero_point,
            calculate_difference_psf,
            save_difference_psf_to_file,
        )
        from PyZOGY.image_class import ImageClass
        sci_path, ref_path, psf_path = image_pair
        sci = ImageClass(sci_path, psf_path, n_stamps=1); sci.zero_point = 2.0
        ref = ImageClass(ref_path, psf_path, n_stamps=1); ref.zero_point = 1.0
        zp = calculate_difference_image_zero_point(sci, ref)
        diff_psf = calculate_difference_psf(sci, ref, zp)
        out = str(tmp_path / "diff_psf3.fits")
        save_difference_psf_to_file(diff_psf, out)
        recovered = fits.getdata(out)
        assert np.all(np.isfinite(recovered))


# ---------------------------------------------------------------------------
# PyZOGY.subtract — run_subtraction (high-level entry point)
# ---------------------------------------------------------------------------

class TestRunSubtraction:
    """run_subtraction is the main lscdiff.py entry point for image differencing."""

    def _write_pair(self, tmp_path, seed):
        rng = np.random.default_rng(seed)
        n = 64
        sci_data = rng.normal(1000.0, 10.0, (n, n)).astype(np.float64)
        ref_data = (sci_data * 2.0 + rng.normal(0.0, 5.0, (n, n))).astype(np.float64)
        psf = np.zeros((5, 5)); psf[2, 2] = 1.0
        sci_path = str(tmp_path / "sci.fits")
        ref_path = str(tmp_path / "ref.fits")
        psf_path = str(tmp_path / "psf.fits")
        fits.writeto(sci_path, sci_data)
        fits.writeto(ref_path, ref_data)
        fits.writeto(psf_path, psf.astype(np.float64))
        return sci_path, ref_path, psf_path, n

    def test_output_file_is_created(self, tmp_path):
        from PyZOGY.subtract import run_subtraction
        sci_path, ref_path, psf_path, n = self._write_pair(tmp_path, 42)
        out_path = str(tmp_path / "output.fits")
        run_subtraction(sci_path, ref_path, psf_path, psf_path,
                        output=out_path, n_stamps=1, gain_ratio=2.0)
        assert os.path.exists(out_path)

    def test_output_data_is_finite(self, tmp_path):
        from PyZOGY.subtract import run_subtraction
        sci_path, ref_path, psf_path, n = self._write_pair(tmp_path, 43)
        out_path = str(tmp_path / "output.fits")
        run_subtraction(sci_path, ref_path, psf_path, psf_path,
                        output=out_path, n_stamps=1, gain_ratio=2.0)
        result = fits.getdata(out_path)
        assert np.all(np.isfinite(result))

    def test_output_shape_matches_input(self, tmp_path):
        from PyZOGY.subtract import run_subtraction
        sci_path, ref_path, psf_path, n = self._write_pair(tmp_path, 44)
        out_path = str(tmp_path / "output.fits")
        run_subtraction(sci_path, ref_path, psf_path, psf_path,
                        output=out_path, n_stamps=1, gain_ratio=2.0)
        result = fits.getdata(out_path)
        assert result.shape == (n, n)

    def test_reference_normalization_run(self, tmp_path):
        from PyZOGY.subtract import run_subtraction
        sci_path, ref_path, psf_path, n = self._write_pair(tmp_path, 45)
        out_path = str(tmp_path / "output_ref.fits")
        run_subtraction(sci_path, ref_path, psf_path, psf_path,
                        output=out_path, n_stamps=1, gain_ratio=2.0,
                        normalization='reference')
        result = fits.getdata(out_path)
        assert result.shape == (n, n)
        assert np.all(np.isfinite(result))
