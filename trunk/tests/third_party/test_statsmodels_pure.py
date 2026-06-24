"""
Tests for statsmodels usage in lcogtsnpipe.

statsmodels is used in PyZOGY.util via `import statsmodels.api as stats`:
- stats.RLM(y, stats.add_constant(x), stats.robust.norms.TukeyBiweight()).fit()
  — Robust Linear Modelling for flux scaling between science and reference images

Tests are offline and use synthetic linear data.

Additional coverage added:
  - TestOLSInference: tvalues, pvalues, bse, fvalue, nobs, conf_int, summary2
  - TestWLS: Weighted Least Squares — weighted_endog, weights kwarg, rsquared
  - TestGLS: Generalised Least Squares — default sigma, param recovery
  - TestQuantReg: Quantile Regression at q=0.5 (median regression)
  - TestRLMNorms: all M-estimator norms available in sm.robust.norms
  - TestRLMWeights: RLM weights attribute, weights shape, outlier down-weighting
  - TestTukeyBiweightNorm: weights(), psi() methods; zero weight beyond tuning constant
  - TestAddConstantEdgeCases: float32 input, 2-D multi-column input, has_constant='add'
  - TestPyZOGYFluxScalingPattern: end-to-end RLM pattern from PyZOGY.util line 252
"""
import numpy as np
import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

class TestStatsmodelsImport:
    def test_statsmodels_api_importable(self):
        import statsmodels.api as sm
        assert hasattr(sm, "OLS")
        assert hasattr(sm, "RLM")
        assert hasattr(sm, "add_constant")

    def test_robust_norms_importable(self):
        import statsmodels.api as sm
        assert hasattr(sm.robust, "norms")
        assert hasattr(sm.robust.norms, "TukeyBiweight")


# ---------------------------------------------------------------------------
# add_constant
# ---------------------------------------------------------------------------

class TestAddConstant:
    def test_adds_intercept_column(self):
        import statsmodels.api as sm
        x = np.array([1.0, 2.0, 3.0])
        X = sm.add_constant(x)
        assert X.shape == (3, 2)
        np.testing.assert_array_equal(X[:, 0], [1.0, 1.0, 1.0])

    def test_does_not_double_add_constant(self):
        import statsmodels.api as sm
        x = np.ones(5)
        X = sm.add_constant(x)
        # A vector of all ones — add_constant with has_constant='raise' would raise,
        # but default should not crash
        assert X.ndim == 2

    def test_2d_input(self):
        import statsmodels.api as sm
        x = np.column_stack([np.arange(5, dtype=float), np.ones(5)])
        X = sm.add_constant(x, has_constant="add")
        assert X.shape[1] == 3


# ---------------------------------------------------------------------------
# OLS — Ordinary Least Squares
# ---------------------------------------------------------------------------

class TestOLS:
    """OLS is the simpler form used before robust fitting."""

    def test_ols_perfect_fit(self):
        import statsmodels.api as sm
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 10, 20)
        y = 3.0 * x + 1.5   # perfect linear relationship
        X = sm.add_constant(x)
        result = sm.OLS(y, X).fit()
        assert result.params[0] == pytest.approx(1.5, abs=1e-8)
        assert result.params[1] == pytest.approx(3.0, abs=1e-8)

    def test_ols_r_squared_perfect(self):
        import statsmodels.api as sm
        x = np.arange(10, dtype=float)
        y = 2.0 * x + 5.0
        result = sm.OLS(y, sm.add_constant(x)).fit()
        assert result.rsquared == pytest.approx(1.0, abs=1e-10)

    def test_ols_residuals_length(self):
        import statsmodels.api as sm
        rng = np.random.default_rng(1)
        x = rng.uniform(0, 5, 30)
        y = 1.5 * x + rng.normal(0, 0.5, 30)
        result = sm.OLS(y, sm.add_constant(x)).fit()
        assert len(result.resid) == 30

    def test_ols_predict(self):
        import statsmodels.api as sm
        x_train = np.arange(10, dtype=float)
        y_train = 2.0 * x_train
        result = sm.OLS(y_train, sm.add_constant(x_train)).fit()
        x_new = sm.add_constant(np.array([11.0, 12.0]))
        preds = result.predict(x_new)
        assert preds[0] == pytest.approx(22.0, abs=0.1)

    def test_ols_params_count(self):
        import statsmodels.api as sm
        x = np.arange(10, dtype=float)
        y = 3.0 * x + 7.0
        result = sm.OLS(y, sm.add_constant(x)).fit()
        assert len(result.params) == 2  # intercept + slope


# ---------------------------------------------------------------------------
# RLM — Robust Linear Model (exact usage in PyZOGY.util line 252)
# ---------------------------------------------------------------------------

class TestRLM:
    """
    Mirrors PyZOGY.util usage:
        stats.RLM(y, stats.add_constant(x), stats.robust.norms.TukeyBiweight()).fit()
    """

    def _fit_rlm(self, x, y):
        import statsmodels.api as sm
        return sm.RLM(y, sm.add_constant(x),
                      M=sm.robust.norms.TukeyBiweight()).fit()

    def test_rlm_fits_clean_data(self):
        rng = np.random.default_rng(5)
        x = rng.uniform(0, 10, 30)
        y = 2.5 * x + 1.0 + rng.normal(0, 0.3, 30)
        result = self._fit_rlm(x, y)
        assert result.params[1] == pytest.approx(2.5, abs=0.3)

    def test_rlm_robust_to_outliers(self):
        """RLM should recover slope despite large outliers."""
        rng = np.random.default_rng(6)
        x = rng.uniform(0, 10, 30)
        y = 2.0 * x + rng.normal(0, 0.2, 30)
        # inject 3 extreme outliers
        y[0] = 500.0
        y[1] = -400.0
        y[2] = 1000.0
        result = self._fit_rlm(x, y)
        # OLS would be badly distorted; RLM should stay close to 2.0
        assert abs(result.params[1] - 2.0) < 1.0

    def test_rlm_returns_params_and_resid(self):
        rng = np.random.default_rng(7)
        x = rng.uniform(1, 5, 20)
        y = x + rng.normal(0, 0.1, 20)
        result = self._fit_rlm(x, y)
        assert len(result.params) == 2
        assert len(result.resid) == 20

    def test_rlm_predict_method(self):
        import statsmodels.api as sm
        x = np.arange(10, dtype=float)
        y = 3.0 * x
        result = self._fit_rlm(x, y)
        # predict expects (n_obs, n_params) shape — add_constant gives (1, 2) for a
        # scalar input which is correct; ensure it's 2-D with at least 2 cols.
        x_new = sm.add_constant(np.array([11.0]), has_constant='add')
        preds = result.predict(x_new)
        assert preds[0] == pytest.approx(33.0, abs=1.0)

    def test_rlm_flux_scaling_pattern(self):
        """
        Simulate the exact PyZOGY use-case: fit a flux scaling ratio between
        matched source fluxes in science and reference images.
        """
        rng = np.random.default_rng(8)
        n = 50
        ref_flux = rng.uniform(100, 10000, n)
        # science flux ≈ 0.8 × ref_flux (different photometric zero-point)
        sci_flux = 0.8 * ref_flux + rng.normal(0, 30, n)
        result = self._fit_rlm(ref_flux, sci_flux)
        # slope should recover ~0.8
        assert result.params[1] == pytest.approx(0.8, abs=0.1)

    def test_tukey_biweight_instantiable(self):
        import statsmodels.api as sm
        norm = sm.robust.norms.TukeyBiweight()
        assert norm is not None


# ---------------------------------------------------------------------------
# OLS inference attributes
# ---------------------------------------------------------------------------

class TestOLSInference:
    """OLS result attributes used for inference and diagnostics."""

    def _fit(self):
        import statsmodels.api as sm
        rng = np.random.default_rng(10)
        x = rng.uniform(0, 10, 30)
        y = 2.5 * x + 1.0 + rng.normal(0, 0.3, 30)
        return sm.OLS(y, sm.add_constant(x)).fit()

    def test_tvalues_shape(self):
        result = self._fit()
        assert result.tvalues.shape == (2,)

    def test_pvalues_in_range(self):
        result = self._fit()
        assert np.all(result.pvalues >= 0)
        assert np.all(result.pvalues <= 1)

    def test_bse_positive(self):
        """Standard errors must be positive."""
        result = self._fit()
        assert np.all(result.bse > 0)

    def test_fvalue_positive(self):
        result = self._fit()
        assert result.fvalue > 0

    def test_nobs(self):
        result = self._fit()
        assert result.nobs == pytest.approx(30)

    def test_conf_int_shape(self):
        result = self._fit()
        ci = result.conf_int()
        assert ci.shape == (2, 2)

    def test_conf_int_lower_lt_upper(self):
        result = self._fit()
        ci = result.conf_int()
        assert np.all(ci[:, 0] < ci[:, 1])

    def test_summary2_has_tables(self):
        result = self._fit()
        s = result.summary2()
        assert len(s.tables) > 0

    def test_rsquared_between_0_and_1(self):
        result = self._fit()
        assert 0.0 <= result.rsquared <= 1.0

    def test_slope_in_confidence_interval(self):
        """True slope (2.5) must lie within the 95% CI of the slope estimate."""
        result = self._fit()
        ci = result.conf_int()
        # ci[1] is the slope row: [lower, upper]
        assert ci[1, 0] < 2.5 < ci[1, 1]


# ---------------------------------------------------------------------------
# WLS — Weighted Least Squares
# ---------------------------------------------------------------------------

class TestWLS:
    """WLS is the precision-weighted variant of OLS."""

    def _fit(self, weights=None):
        import statsmodels.api as sm
        rng = np.random.default_rng(20)
        x = rng.uniform(0, 10, 40)
        y = 3.0 * x + 2.0 + rng.normal(0, 0.5, 40)
        if weights is None:
            weights = np.ones(40)
        return sm.WLS(y, sm.add_constant(x), weights=weights).fit()

    def test_wls_uniform_weights_matches_ols(self):
        """With equal weights, WLS slope should equal OLS slope."""
        import statsmodels.api as sm
        rng = np.random.default_rng(21)
        x = rng.uniform(0, 10, 40)
        y = 3.0 * x + 2.0 + rng.normal(0, 0.5, 40)
        X = sm.add_constant(x)
        ols = sm.OLS(y, X).fit()
        wls = sm.WLS(y, X, weights=np.ones(40)).fit()
        assert wls.params[1] == pytest.approx(ols.params[1], rel=1e-8)

    def test_wls_rsquared_positive(self):
        result = self._fit()
        assert result.rsquared > 0

    def test_wls_params_count(self):
        result = self._fit()
        assert len(result.params) == 2

    def test_wls_slope_recovery(self):
        result = self._fit()
        assert result.params[1] == pytest.approx(3.0, abs=0.3)

    def test_wls_non_uniform_weights_accepted(self):
        rng = np.random.default_rng(22)
        weights = rng.uniform(0.1, 5.0, 40)
        result = self._fit(weights=weights)
        assert isinstance(result.params[1], float)

    def test_wls_residuals_length(self):
        result = self._fit()
        assert len(result.resid) == 40


# ---------------------------------------------------------------------------
# GLS — Generalised Least Squares
# ---------------------------------------------------------------------------

class TestGLS:
    """GLS with identity sigma is equivalent to OLS."""

    def test_gls_default_sigma_recovers_slope(self):
        import statsmodels.api as sm
        rng = np.random.default_rng(30)
        x = rng.uniform(0, 10, 25)
        y = 4.0 * x + 0.5 + rng.normal(0, 0.4, 25)
        result = sm.GLS(y, sm.add_constant(x)).fit()
        assert result.params[1] == pytest.approx(4.0, abs=0.4)

    def test_gls_params_count(self):
        import statsmodels.api as sm
        x = np.arange(15, dtype=float)
        y = 1.5 * x + 3.0
        result = sm.GLS(y, sm.add_constant(x)).fit()
        assert len(result.params) == 2

    def test_gls_resid_finite(self):
        import statsmodels.api as sm
        rng = np.random.default_rng(31)
        x = rng.uniform(0, 5, 20)
        y = x + rng.normal(0, 0.2, 20)
        result = sm.GLS(y, sm.add_constant(x)).fit()
        assert np.all(np.isfinite(result.resid))


# ---------------------------------------------------------------------------
# QuantReg — Quantile Regression
# ---------------------------------------------------------------------------

class TestQuantReg:
    """QuantReg at q=0.5 gives median regression — robust to outliers."""

    def _fit(self, q=0.5):
        import statsmodels.api as sm
        rng = np.random.default_rng(40)
        x = rng.uniform(0, 10, 40)
        y = 2.0 * x + 1.5 + rng.normal(0, 0.5, 40)
        return sm.QuantReg(y, sm.add_constant(x)).fit(q=q)

    def test_quantreg_slope_recovery(self):
        result = self._fit(q=0.5)
        assert result.params[1] == pytest.approx(2.0, abs=0.3)

    def test_quantreg_params_count(self):
        result = self._fit()
        assert len(result.params) == 2

    def test_quantreg_q_90_slope_positive(self):
        result = self._fit(q=0.9)
        assert result.params[1] > 0

    def test_quantreg_residuals_length(self):
        result = self._fit()
        assert len(result.resid) == 40

    def test_quantreg_predict(self):
        import statsmodels.api as sm
        result = self._fit()
        x_new = sm.add_constant(np.array([5.0]), has_constant='add')
        preds = result.predict(x_new)
        assert preds[0] == pytest.approx(2.0 * 5.0 + 1.5, abs=1.0)


# ---------------------------------------------------------------------------
# RLM M-estimator norms
# ---------------------------------------------------------------------------

class TestRLMNorms:
    """All M-estimator norm classes available in sm.robust.norms."""

    def _fit_with_norm(self, norm_instance):
        import statsmodels.api as sm
        rng = np.random.default_rng(50)
        x = rng.uniform(0, 10, 30)
        y = 2.5 * x + 1.0 + rng.normal(0, 0.3, 30)
        return sm.RLM(y, sm.add_constant(x), M=norm_instance).fit()

    def test_hubert_norm(self):
        import statsmodels.api as sm
        result = self._fit_with_norm(sm.robust.norms.HuberT())
        assert result.params[1] == pytest.approx(2.5, abs=0.3)

    def test_andrew_wave_norm(self):
        import statsmodels.api as sm
        result = self._fit_with_norm(sm.robust.norms.AndrewWave())
        assert result.params[1] == pytest.approx(2.5, abs=0.3)

    def test_hampel_norm(self):
        import statsmodels.api as sm
        result = self._fit_with_norm(sm.robust.norms.Hampel())
        assert result.params[1] == pytest.approx(2.5, abs=0.3)

    def test_least_squares_norm_matches_ols(self):
        """LeastSquares norm turns RLM into plain OLS."""
        import statsmodels.api as sm
        rng = np.random.default_rng(51)
        x = rng.uniform(0, 10, 30)
        y = 2.5 * x + 1.0 + rng.normal(0, 0.3, 30)
        X = sm.add_constant(x)
        ols = sm.OLS(y, X).fit()
        rlm_ls = sm.RLM(y, X, M=sm.robust.norms.LeastSquares()).fit()
        assert rlm_ls.params[1] == pytest.approx(ols.params[1], rel=1e-5)

    def test_ramsay_norm(self):
        import statsmodels.api as sm
        result = self._fit_with_norm(sm.robust.norms.RamsayE())
        assert result.params[1] == pytest.approx(2.5, abs=0.3)

    def test_trimmed_mean_norm(self):
        import statsmodels.api as sm
        result = self._fit_with_norm(sm.robust.norms.TrimmedMean())
        assert result.params[1] == pytest.approx(2.5, abs=0.5)

    def test_all_norms_importable(self):
        import statsmodels.api as sm
        norm_names = ['HuberT', 'AndrewWave', 'Hampel', 'LeastSquares',
                      'RamsayE', 'TrimmedMean', 'TukeyBiweight']
        for name in norm_names:
            assert hasattr(sm.robust.norms, name), f'{name} not found'


# ---------------------------------------------------------------------------
# RLM weights attribute
# ---------------------------------------------------------------------------

class TestRLMWeights:
    """RLM.fit() produces a weights attribute that down-weights outliers."""

    def _fit_with_outliers(self):
        import statsmodels.api as sm
        rng = np.random.default_rng(60)
        x = rng.uniform(0, 10, 40)
        y = 2.0 * x + rng.normal(0, 0.2, 40)
        # inject 4 extreme outliers
        y[0] = 500.0
        y[1] = -400.0
        y[2] = 600.0
        y[3] = -350.0
        return sm.RLM(y, sm.add_constant(x),
                      M=sm.robust.norms.TukeyBiweight()).fit()

    def test_weights_shape(self):
        result = self._fit_with_outliers()
        assert result.weights.shape == (40,)

    def test_weights_non_negative(self):
        result = self._fit_with_outliers()
        assert np.all(result.weights >= 0)

    def test_outliers_downweighted(self):
        """The 4 injected outliers should have near-zero weight."""
        result = self._fit_with_outliers()
        assert result.weights[0] < 0.1
        assert result.weights[1] < 0.1

    def test_inliers_have_higher_weight_than_outliers(self):
        result = self._fit_with_outliers()
        mean_inlier_weight = np.mean(result.weights[4:])
        assert mean_inlier_weight > result.weights[0]
        assert mean_inlier_weight > result.weights[1]


# ---------------------------------------------------------------------------
# TukeyBiweight norm methods
# ---------------------------------------------------------------------------

class TestTukeyBiweightNorm:
    """TukeyBiweight norm function methods."""

    def test_weights_near_zero_gives_one(self):
        import statsmodels.api as sm
        norm = sm.robust.norms.TukeyBiweight()
        w = norm.weights(np.array([0.0]))
        assert w[0] == pytest.approx(1.0, abs=1e-8)

    def test_weights_beyond_tuning_constant_is_zero(self):
        """Default c=4.685; z=100 is way outside, so weight=0."""
        import statsmodels.api as sm
        norm = sm.robust.norms.TukeyBiweight()
        w = norm.weights(np.array([100.0]))
        assert w[0] == pytest.approx(0.0, abs=1e-8)

    def test_psi_zero_at_zero(self):
        import statsmodels.api as sm
        norm = sm.robust.norms.TukeyBiweight()
        p = norm.psi(np.array([0.0]))
        assert p[0] == pytest.approx(0.0, abs=1e-8)

    def test_weights_decreasing_with_magnitude(self):
        """TukeyBiweight weights must decrease as |z| increases (within cutoff)."""
        import statsmodels.api as sm
        norm = sm.robust.norms.TukeyBiweight()
        z = np.array([0.0, 1.0, 2.0, 3.0])
        w = norm.weights(z)
        assert w[0] >= w[1] >= w[2] >= w[3]

    def test_custom_tuning_constant(self):
        import statsmodels.api as sm
        norm = sm.robust.norms.TukeyBiweight(c=6.0)
        # At z=5.0, still inside c=6 — weight should be positive
        w = norm.weights(np.array([5.0]))
        assert w[0] > 0


# ---------------------------------------------------------------------------
# add_constant edge cases
# ---------------------------------------------------------------------------

class TestAddConstantEdgeCases:
    """Edge cases for add_constant not covered by the basic tests."""

    def test_float32_input_accepted(self):
        import statsmodels.api as sm
        x = np.arange(6, dtype=np.float32)
        X = sm.add_constant(x)
        assert X.shape == (6, 2)

    def test_integer_input_accepted(self):
        import statsmodels.api as sm
        x = np.arange(5)
        X = sm.add_constant(x)
        assert X.shape[1] == 2

    def test_multi_column_adds_one_column(self):
        import statsmodels.api as sm
        x = np.column_stack([np.arange(8, dtype=float),
                              np.arange(8, dtype=float) ** 2])
        X = sm.add_constant(x)
        assert X.shape == (8, 3)

    def test_constant_column_is_first(self):
        import statsmodels.api as sm
        x = np.arange(4, dtype=float) + 1.0
        X = sm.add_constant(x)
        np.testing.assert_array_equal(X[:, 0], np.ones(4))

    def test_has_constant_add_forces_column(self):
        """has_constant='add' always adds the column even if data already has one."""
        import statsmodels.api as sm
        x = np.column_stack([np.ones(5), np.arange(5, dtype=float)])
        X = sm.add_constant(x, has_constant='add')
        # result should have 3 columns (original 2 + new constant)
        assert X.shape[1] == 3


# ---------------------------------------------------------------------------
# PyZOGY.util flux-scaling end-to-end pattern (line 252)
# ---------------------------------------------------------------------------

class TestPyZOGYFluxScalingPattern:
    """
    Reproduce the exact statsmodels call from PyZOGY.util:

        stats.RLM(
            sci_flux,
            stats.add_constant(ref_flux),
            M=stats.robust.norms.TukeyBiweight()
        ).fit()

    Used to compute the flux ratio (zero-point offset) between science
    and reference images from matched point-source fluxes.
    """

    def _make_matched_fluxes(self, n=60, ratio=0.75, seed=0):
        rng = np.random.default_rng(seed)
        ref_flux = rng.uniform(500, 50000, n)
        sci_flux = ratio * ref_flux + rng.normal(0, 50, n)
        return ref_flux, sci_flux

    def test_pyzogy_rlm_recovers_flux_ratio(self):
        import statsmodels.api as sm
        ref_flux, sci_flux = self._make_matched_fluxes(ratio=0.75)
        result = sm.RLM(
            sci_flux,
            sm.add_constant(ref_flux),
            M=sm.robust.norms.TukeyBiweight(),
        ).fit()
        assert result.params[1] == pytest.approx(0.75, abs=0.05)

    def test_pyzogy_rlm_with_heavy_outliers(self):
        """Flux ratio recovery must survive 10% bad measurements."""
        import statsmodels.api as sm
        rng = np.random.default_rng(1)
        n = 80
        ref_flux = rng.uniform(500, 50000, n)
        sci_flux = 1.2 * ref_flux + rng.normal(0, 50, n)
        # inject 8 extreme bad matches
        sci_flux[:8] = rng.uniform(1e5, 5e5, 8)
        result = sm.RLM(
            sci_flux,
            sm.add_constant(ref_flux),
            M=sm.robust.norms.TukeyBiweight(),
        ).fit()
        assert result.params[1] == pytest.approx(1.2, abs=0.2)

    def test_pyzogy_rlm_slope_is_in_params(self):
        import statsmodels.api as sm
        ref_flux, sci_flux = self._make_matched_fluxes()
        result = sm.RLM(
            sci_flux,
            sm.add_constant(ref_flux),
            M=sm.robust.norms.TukeyBiweight(),
        ).fit()
        # params[0] = intercept, params[1] = slope
        assert len(result.params) == 2

    def test_pyzogy_rlm_weights_shape_matches_n(self):
        import statsmodels.api as sm
        ref_flux, sci_flux = self._make_matched_fluxes(n=50)
        result = sm.RLM(
            sci_flux,
            sm.add_constant(ref_flux),
            M=sm.robust.norms.TukeyBiweight(),
        ).fit()
        assert result.weights.shape == (50,)

    def test_pyzogy_rlm_residuals_finite(self):
        import statsmodels.api as sm
        ref_flux, sci_flux = self._make_matched_fluxes(n=40, seed=5)
        result = sm.RLM(
            sci_flux,
            sm.add_constant(ref_flux),
            M=sm.robust.norms.TukeyBiweight(),
        ).fit()
        assert np.all(np.isfinite(result.resid))
