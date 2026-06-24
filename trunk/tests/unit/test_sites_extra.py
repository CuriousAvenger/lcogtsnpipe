"""
Additional tests for lsc.sites — covers chosecolor edge cases and
filter mapping completeness.
"""
import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# chosecolor — additional edge cases
# ---------------------------------------------------------------------------

class TestChosecolorExtra:
    def test_empty_filter_list(self):
        from lsc.sites import chosecolor
        result = chosecolor([])
        assert result == {}

    def test_single_filter(self):
        from lsc.sites import chosecolor
        result = chosecolor(['B'])
        assert 'B' in result
        assert result['B'] == []

    def test_usegood_prefers_primary_color(self):
        """When usegood=True, each filter should get its primary color."""
        from lsc.sites import chosecolor
        result = chosecolor(['U', 'B', 'V'], usegood=True)
        # U should map to UB, B to BV, V to VR
        assert 'UB' in result['U']
        assert 'BV' in result['B']
        assert 'VR' in result['V']

    def test_usegood_with_incomplete_filters(self):
        """Filters without a primary color entry still work."""
        from lsc.sites import chosecolor
        # 'z' has no goodcol entry
        result = chosecolor(['z', 'i'], usegood=True)
        assert 'z' in result
        assert 'i' in result

    def test_all_filters(self):
        """Test with a realistic full filter set."""
        from lsc.sites import chosecolor
        result = chosecolor(['U', 'B', 'V', 'R', 'I'])
        # All should have at least one color
        for filt in ['U', 'B', 'V', 'R', 'I']:
            assert filt in result
            assert len(result[filt]) > 0

    def test_sloan_filters(self):
        from lsc.sites import chosecolor
        result = chosecolor(['u', 'g', 'r', 'i', 'z'])
        for filt in ['u', 'g', 'r', 'i', 'z']:
            assert filt in result


# ---------------------------------------------------------------------------
# filterst / filterst1 — filter name mappings
# ---------------------------------------------------------------------------

class TestFilterMappings:
    def test_filterst1_reverse_mapping(self):
        """filterst1 should map instrument filter names to canonical names."""
        from lsc.sites import filterst1
        # SDSS mappings
        assert filterst1['up'] == 'u'
        assert filterst1['gp'] == 'g'
        assert filterst1['rp'] == 'r'
        assert filterst1['ip'] == 'i'
        assert filterst1['zs'] == 'z'

    def test_filterst_landolt_completeness(self):
        """Landolt system should include UBVRI variants."""
        from lsc.sites import filterst
        for name in ['U', 'B', 'V', 'R', 'I']:
            assert name in filterst['landolt']

    def test_filterst_sloan_completeness(self):
        """Sloan system should include ugriz variants."""
        from lsc.sites import filterst
        for name in ['u', 'g', 'r', 'i', 'z']:
            assert name in filterst['sloan']

    def test_filterst_apass_completeness(self):
        """APASS system should include BVgri variants."""
        from lsc.sites import filterst
        for name in ['B', 'V', 'g', 'r', 'i']:
            assert name in filterst['apass']

    def test_extinction_has_common_sites(self):
        """Extinction dict should have entries for common sites."""
        from lsc.sites import extinction
        for site in ['lsc', 'coj', 'ogg', 'elp', 'cpt', 'tfn']:
            assert site in extinction

    def test_extinction_has_common_filters(self):
        """Each site should have entries for common filters."""
        from lsc.sites import extinction
        common_filters = ['B', 'V', 'R', 'I', 'g', 'r', 'i']
        for site in extinction:
            for filt in common_filters:
                assert filt in extinction[site], f"Missing {filt} for site {site}"

    def test_ps1_extinction_matches_ogg(self):
        """PS1 extinction should be the same as OGG."""
        from lsc.sites import extinction
        assert extinction['PS1'] == extinction['ogg']

    def test_sdss_extinction_matches_elp(self):
        """SDSS extinction should be the same as ELP."""
        from lsc.sites import extinction
        assert extinction['SDSS'] == extinction['elp']
