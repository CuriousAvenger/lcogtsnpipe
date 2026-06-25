"""
Comprehensive tests for lsc.sites — extinction tables, filter mappings,
chosecolor logic, and edge cases.
"""
import importlib.util
import os
import sys
import pytest

pytestmark = pytest.mark.unit

# Load lsc.sites directly from its file path to avoid triggering lsc/__init__.py
# which imports modules requiring unavailable dependencies (astroquery).
_SITES_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'src', 'lsc', 'sites.py'
)
_spec = importlib.util.spec_from_file_location("lsc.sites", os.path.abspath(_SITES_PATH))
_sites_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sites_mod)
sys.modules.setdefault("lsc.sites", _sites_mod)


# ---------------------------------------------------------------------------
# Extinction table structure and values
# ---------------------------------------------------------------------------

class TestExtinctionTableStructure:
    """Verify the extinction dictionary has correct structure and values."""

    def test_all_observatory_sites_present(self):
        from lsc.sites import extinction
        expected = {'lsc', 'coj', 'ogg', 'elp', 'cpt', 'tfn', None, 'PS1', 'SDSS'}
        assert expected == set(extinction.keys())

    def test_all_sites_have_identical_filter_keys(self):
        """Every site must define the exact same set of filter bands."""
        from lsc.sites import extinction
        reference = set(extinction['lsc'].keys())
        for site, bands in extinction.items():
            assert set(bands.keys()) == reference, (
                f"Site {site!r} bands differ: missing={reference - set(bands.keys())}, "
                f"extra={set(bands.keys()) - reference}"
            )

    def test_expected_filters_in_each_site(self):
        """Each site should contain standard UBVRI + ugriz + JHK + w."""
        from lsc.sites import extinction
        expected_filters = set('UuBgVrRiIzJHKw')
        for site in extinction:
            assert expected_filters == set(extinction[site].keys()), (
                f"Site {site!r} missing filters"
            )

    def test_all_values_are_non_negative_floats(self):
        from lsc.sites import extinction
        for site, bands in extinction.items():
            for band, val in bands.items():
                assert isinstance(val, float), f"{site}/{band} is {type(val)}"
                assert val >= 0.0, f"Negative extinction at {site}/{band}: {val}"

    def test_extinction_values_within_reasonable_range(self):
        """Extinction coefficients should typically be < 1.0 mag/airmass."""
        from lsc.sites import extinction
        for site, bands in extinction.items():
            for band, val in bands.items():
                assert val < 2.0, (
                    f"Unreasonably high extinction at {site}/{band}: {val}"
                )

    def test_uv_extinction_higher_than_infrared(self):
        """For all real sites, U/u-band extinction should exceed I/i-band."""
        from lsc.sites import extinction
        for site in ['lsc', 'coj', 'ogg', 'elp', 'cpt', 'tfn']:
            assert extinction[site]['U'] > extinction[site]['I'], (
                f"Site {site}: U ({extinction[site]['U']}) not > I ({extinction[site]['I']})"
            )
            assert extinction[site]['u'] > extinction[site]['i'], (
                f"Site {site}: u ({extinction[site]['u']}) not > i ({extinction[site]['i']})"
            )


class TestExtinctionAliases:
    """Test PS1 and SDSS alias entries."""

    def test_ps1_is_same_object_as_ogg(self):
        from lsc.sites import extinction
        assert extinction['PS1'] is extinction['ogg']

    def test_sdss_is_same_object_as_elp(self):
        from lsc.sites import extinction
        assert extinction['SDSS'] is extinction['elp']

    def test_ps1_values_match_ogg(self):
        from lsc.sites import extinction
        for band in extinction['ogg']:
            assert extinction['PS1'][band] == extinction['ogg'][band]

    def test_sdss_values_match_elp(self):
        from lsc.sites import extinction
        for band in extinction['elp']:
            assert extinction['SDSS'][band] == extinction['elp'][band]

    def test_modifying_ps1_modifies_ogg(self):
        """Since PS1 is ogg by reference, verify the alias semantics."""
        from lsc.sites import extinction
        # They are the same object
        assert id(extinction['PS1']) == id(extinction['ogg'])

    def test_modifying_sdss_modifies_elp(self):
        from lsc.sites import extinction
        assert id(extinction['SDSS']) == id(extinction['elp'])


class TestExtinctionDefaultSite:
    """Test the None (default/fallback) site."""

    def test_none_site_exists(self):
        from lsc.sites import extinction
        assert None in extinction

    def test_none_site_has_all_filters(self):
        from lsc.sites import extinction
        expected_filters = set('UuBgVrRiIzJHKw')
        assert set(extinction[None].keys()) == expected_filters

    def test_none_site_values_are_averages(self):
        """None site values should be roughly average of other sites."""
        from lsc.sites import extinction
        # Just verify they're reasonable (between min and max of other sites)
        real_sites = ['lsc', 'coj', 'ogg', 'elp', 'cpt', 'tfn']
        for band in extinction[None]:
            vals = [extinction[s][band] for s in real_sites]
            # The None default should be within the range of real sites
            # (or at least reasonably close)
            assert extinction[None][band] <= max(vals) + 0.1
            assert extinction[None][band] >= min(vals) - 0.1


# ---------------------------------------------------------------------------
# filterst — filter name mappings
# ---------------------------------------------------------------------------

class TestFilterst:
    """Test the filterst dictionary structure and content."""

    def test_canonical_bands_present(self):
        from lsc.sites import filterst
        for band in 'UBVRIugrizw':
            assert band in filterst

    def test_each_band_has_nonempty_list(self):
        from lsc.sites import filterst
        for band in 'UBVRIugrizw':
            assert isinstance(filterst[band], list)
            assert len(filterst[band]) >= 1

    def test_canonical_band_name_is_first_alias(self):
        """The first alias in the list should be the instrument filter name."""
        from lsc.sites import filterst
        # Verify known first entries
        assert filterst['U'][0] == 'U'
        assert filterst['B'][0] == 'B'
        assert filterst['V'][0] == 'V'
        assert filterst['R'][0] == 'R'
        assert filterst['I'][0] == 'I'
        assert filterst['u'][0] == 'up'
        assert filterst['g'][0] == 'gp'
        assert filterst['r'][0] == 'rp'
        assert filterst['i'][0] == 'ip'
        assert filterst['z'][0] == 'zs'
        assert filterst['w'][0] == 'w'

    def test_sdss_aliases(self):
        """Verify SDSS-style aliases (e.g., SDSS-U, SDSS-G) are present."""
        from lsc.sites import filterst
        assert 'SDSS-U' in filterst['u']
        assert 'SDSS-G' in filterst['g']
        assert 'SDSS-R' in filterst['r']
        assert 'SDSS-I' in filterst['i']

    def test_bessell_aliases(self):
        """Verify Bessell-style aliases are present."""
        from lsc.sites import filterst
        assert 'Bessell-B' in filterst['B']
        assert 'Bessell-V' in filterst['V']
        assert 'Bessell-R' in filterst['R']
        assert 'Bessell-I' in filterst['I']

    def test_astrodon_u_alias(self):
        from lsc.sites import filterst
        assert 'Astrodon-U' in filterst['U']

    def test_pan_starrs_aliases(self):
        from lsc.sites import filterst
        assert 'Pan-Starrs' in filterst['z']
        assert 'Pan-Starrs-Z' in filterst['z']

    def test_w_band_has_w_alias(self):
        from lsc.sites import filterst
        assert 'w' in filterst['w']
        assert len(filterst['w']) == 1


class TestFilterstGroups:
    """Test the composite filter groups (landolt, sloan, apass, gaia, empty)."""

    def test_landolt_group_has_ubvri_aliases(self):
        from lsc.sites import filterst
        for band in 'UBVRI':
            for alias in filterst[band]:
                assert alias in filterst['landolt']

    def test_sloan_group_has_ugrizw_aliases(self):
        from lsc.sites import filterst
        for band in 'ugrizw':
            for alias in filterst[band]:
                assert alias in filterst['sloan']

    def test_apass_group_has_bvgriw_aliases(self):
        from lsc.sites import filterst
        for band in 'BVgriw':
            for alias in filterst[band]:
                assert alias in filterst['apass']

    def test_gaia_group_contains_all_bands(self):
        from lsc.sites import filterst
        for band in 'UBVRIugrizw':
            for alias in filterst[band]:
                assert alias in filterst['gaia']

    def test_empty_string_group_equals_landolt_plus_sloan(self):
        from lsc.sites import filterst
        combined = set(filterst['landolt'] + filterst['sloan'])
        assert set(filterst['']) == combined

    def test_landolt_does_not_overlap_with_sloan(self):
        """Landolt and Sloan groups should have no common aliases."""
        from lsc.sites import filterst
        landolt_set = set(filterst['landolt'])
        sloan_set = set(filterst['sloan'])
        # Some may overlap (e.g., w is in both apass and sloan via 'w')
        # But landolt (UBVRI) and sloan (ugrizw) should be disjoint
        assert landolt_set.isdisjoint(sloan_set)

    def test_gaia_is_superset_of_all_groups(self):
        from lsc.sites import filterst
        gaia_set = set(filterst['gaia'])
        assert set(filterst['landolt']).issubset(gaia_set)
        assert set(filterst['sloan']).issubset(gaia_set)
        assert set(filterst['apass']).issubset(gaia_set)


class TestFilterst1:
    """Test the reverse mapping filterst1."""

    def test_all_aliases_mapped_back(self):
        from lsc.sites import filterst, filterst1
        for canonical, aliases in filterst.items():
            if isinstance(canonical, str) and len(canonical) == 1:
                for alias in aliases:
                    assert alias in filterst1
                    assert filterst1[alias] == canonical

    def test_no_duplicate_aliases_across_bands(self):
        """No alias should appear in two different canonical band lists."""
        from lsc.sites import filterst
        seen = {}
        for canonical, aliases in filterst.items():
            if isinstance(canonical, str) and len(canonical) == 1:
                for alias in aliases:
                    assert alias not in seen, (
                        f"Alias '{alias}' found in both '{seen[alias]}' and '{canonical}'"
                    )
                    seen[alias] = canonical

    def test_specific_reverse_mappings(self):
        from lsc.sites import filterst1
        assert filterst1['up'] == 'u'
        assert filterst1['gp'] == 'g'
        assert filterst1['rp'] == 'r'
        assert filterst1['ip'] == 'i'
        assert filterst1['zs'] == 'z'
        assert filterst1['U'] == 'U'
        assert filterst1['B'] == 'B'
        assert filterst1['V'] == 'V'
        assert filterst1['R'] == 'R'
        assert filterst1['I'] == 'I'
        assert filterst1['w'] == 'w'

    def test_filterst1_keys_count(self):
        """filterst1 should have exactly as many keys as total aliases."""
        from lsc.sites import filterst, filterst1
        total_aliases = sum(
            len(v) for k, v in filterst.items()
            if isinstance(k, str) and len(k) == 1
        )
        assert len(filterst1) == total_aliases

    def test_filterst1_does_not_contain_group_names(self):
        """Group names (landolt, sloan, etc.) should not be in filterst1."""
        from lsc.sites import filterst1
        assert 'landolt' not in filterst1
        assert 'sloan' not in filterst1
        assert 'apass' not in filterst1
        assert 'gaia' not in filterst1


# ---------------------------------------------------------------------------
# chosecolor — color pair selection logic
# ---------------------------------------------------------------------------

class TestChosecolorBasic:
    """Test basic chosecolor behavior."""

    def test_returns_dict(self):
        from lsc.sites import chosecolor
        result = chosecolor(['B', 'V'])
        assert isinstance(result, dict)

    def test_keys_match_input_filters(self):
        from lsc.sites import chosecolor
        filters = ['B', 'V', 'R']
        result = chosecolor(filters)
        assert set(result.keys()) == set(filters)

    def test_empty_input_returns_empty_dict(self):
        from lsc.sites import chosecolor
        result = chosecolor([])
        assert result == {}

    def test_single_filter_no_pairs(self):
        from lsc.sites import chosecolor
        for filt in 'UBVRIugriz':
            result = chosecolor([filt])
            assert result[filt] == [], f"Filter '{filt}' should have no pairs alone"

    def test_bv_pair(self):
        from lsc.sites import chosecolor
        result = chosecolor(['B', 'V'])
        assert 'BV' in result['B']
        assert 'BV' in result['V']

    def test_ub_pair(self):
        from lsc.sites import chosecolor
        result = chosecolor(['U', 'B'])
        assert 'UB' in result['U']
        assert 'UB' in result['B']

    def test_vr_pair(self):
        from lsc.sites import chosecolor
        result = chosecolor(['V', 'R'])
        assert 'VR' in result['V']
        assert 'VR' in result['R']

    def test_ri_pair(self):
        from lsc.sites import chosecolor
        result = chosecolor(['R', 'I'])
        assert 'RI' in result['R']
        assert 'RI' in result['I']

    def test_ug_pair(self):
        from lsc.sites import chosecolor
        result = chosecolor(['u', 'g'])
        assert 'ug' in result['u']
        assert 'ug' in result['g']

    def test_gr_pair(self):
        from lsc.sites import chosecolor
        result = chosecolor(['g', 'r'])
        assert 'gr' in result['g']
        assert 'gr' in result['r']

    def test_ri_sloan_pair(self):
        from lsc.sites import chosecolor
        result = chosecolor(['r', 'i'])
        assert 'ri' in result['r']
        assert 'ri' in result['i']

    def test_iz_pair(self):
        from lsc.sites import chosecolor
        result = chosecolor(['i', 'z'])
        assert 'iz' in result['i']
        assert 'iz' in result['z']


class TestChosecolorFullSets:
    """Test chosecolor with complete filter sets."""

    def test_full_landolt_set(self):
        from lsc.sites import chosecolor
        result = chosecolor(['U', 'B', 'V', 'R', 'I'])
        # U should have UB
        assert 'UB' in result['U']
        # B should have UB and BV
        assert 'UB' in result['B']
        assert 'BV' in result['B']
        # V should have BV and VR
        assert 'BV' in result['V']
        assert 'VR' in result['V']
        # R should have VR and RI
        assert 'VR' in result['R']
        assert 'RI' in result['R']
        # I should have RI
        assert 'RI' in result['I']

    def test_full_sloan_set(self):
        from lsc.sites import chosecolor
        result = chosecolor(['u', 'g', 'r', 'i', 'z'])
        assert 'ug' in result['u']
        assert 'ug' in result['g']
        assert 'gr' in result['g']
        assert 'gr' in result['r']
        assert 'ri' in result['r']
        assert 'ri' in result['i']
        assert 'iz' in result['i']
        assert 'iz' in result['z']

    def test_mixed_landolt_sloan(self):
        """Mixing Landolt and Sloan filters should still only pair adjacent bands."""
        from lsc.sites import chosecolor
        result = chosecolor(['B', 'g', 'V', 'r'])
        # BV should form since both B and V present
        assert 'BV' in result['B']
        assert 'BV' in result['V']
        # gr should form since both g and r present
        assert 'gr' in result['g']
        assert 'gr' in result['r']


class TestChosecolorUsegood:
    """Test chosecolor with usegood=True."""

    def test_usegood_collapses_to_single_pair(self):
        from lsc.sites import chosecolor
        result = chosecolor(['U', 'B', 'V', 'R', 'I'], usegood=True)
        # Each filter should have exactly one preferred pair
        assert result['U'] == ['UB']
        assert result['B'] == ['BV']
        assert result['V'] == ['VR']
        assert result['R'] == ['VR']
        assert result['I'] == ['RI']

    def test_usegood_sloan(self):
        from lsc.sites import chosecolor
        result = chosecolor(['u', 'g', 'r', 'i', 'z'], usegood=True)
        assert result['u'] == ['ug']
        assert result['g'] == ['gr']
        assert result['r'] == ['ri']
        assert result['i'] == ['ri']
        assert result['z'] == ['iz']

    def test_usegood_when_preferred_unavailable(self):
        """If the preferred pair can't form, fall back to available pairs."""
        from lsc.sites import chosecolor
        # U prefers UB, but B is not available
        result = chosecolor(['U'], usegood=True)
        assert result['U'] == []

    def test_usegood_partial_set(self):
        """With usegood but only some filters available."""
        from lsc.sites import chosecolor
        # V prefers VR but R is missing
        result = chosecolor(['B', 'V'], usegood=True)
        assert result['V'] == ['BV']
        assert result['B'] == ['BV']

    def test_usegood_preserves_available_when_preferred_not_possible(self):
        """If preferred color not in available pairs, keep what's available."""
        from lsc.sites import chosecolor
        # R prefers VR; V not available, I available
        result = chosecolor(['R', 'I'], usegood=True)
        # goodcol['R'] = 'VR' which is NOT in color['R'] (no V present)
        # So it keeps whatever pairs are available
        assert 'RI' in result['R']


class TestChosecolorEdgeCases:
    """Test chosecolor with unusual/edge case inputs."""

    def test_unknown_filter_names(self):
        from lsc.sites import chosecolor
        result = chosecolor(['X', 'Y', 'Z'])
        assert result == {'X': [], 'Y': [], 'Z': []}

    def test_duplicate_filters_in_input(self):
        from lsc.sites import chosecolor
        # Duplicates should not cause errors
        result = chosecolor(['B', 'B', 'V'])
        assert 'B' in result
        assert 'V' in result

    def test_order_independence(self):
        """Result should be the same regardless of input order."""
        from lsc.sites import chosecolor
        r1 = chosecolor(['B', 'V', 'R'])
        r2 = chosecolor(['R', 'V', 'B'])
        # Same keys
        assert set(r1.keys()) == set(r2.keys())
        # Same pairs (as sets for order-independent comparison)
        for k in r1:
            assert set(r1[k]) == set(r2[k])

    def test_w_filter_has_no_color_pairs(self):
        """The 'w' filter has no adjacent pairs defined."""
        from lsc.sites import chosecolor
        result = chosecolor(['w', 'B', 'V'])
        # 'w' is not part of any known color pair
        assert result['w'] == []

    def test_numeric_or_special_characters(self):
        """Non-standard filter names should not crash."""
        from lsc.sites import chosecolor
        result = chosecolor(['1', '2', '#', ''])
        # Should return dict with empty lists
        for k in ['1', '2', '#', '']:
            assert result[k] == []

    def test_very_large_filter_list(self):
        """A large input should still work correctly."""
        from lsc.sites import chosecolor
        big_list = list('UBVRIugriz') * 10
        result = chosecolor(big_list)
        # Should have all unique filters as keys
        assert 'B' in result
        assert 'V' in result

    def test_usegood_false_is_default_behavior(self):
        from lsc.sites import chosecolor
        r1 = chosecolor(['B', 'V', 'R'])
        r2 = chosecolor(['B', 'V', 'R'], usegood=False)
        assert r1 == r2

    def test_color_pair_symmetry(self):
        """If pair XY is in color[X], it should also be in color[Y]."""
        from lsc.sites import chosecolor
        result = chosecolor(['U', 'B', 'V', 'R', 'I', 'u', 'g', 'r', 'i', 'z'])
        all_pairs = ['UB', 'BV', 'VR', 'RI', 'ug', 'gr', 'ri', 'iz']
        for pair in all_pairs:
            f1, f2 = pair[0], pair[1]
            assert pair in result[f1], f"Pair {pair} not in result[{f1}]"
            assert pair in result[f2], f"Pair {pair} not in result[{f2}]"


# ---------------------------------------------------------------------------
# Cross-module consistency
# ---------------------------------------------------------------------------

class TestCrossConsistency:
    """Verify consistency between extinction, filterst, and filterst1."""

    def test_extinction_bands_subset_of_filterst(self):
        """Every band in extinction should be a canonical band in filterst or a known NIR band."""
        from lsc.sites import extinction, filterst
        # Get canonical bands from filterst (single char keys)
        canonical_bands = {k for k in filterst.keys() if isinstance(k, str) and len(k) == 1}
        # J, H, K are in extinction but not in filterst (NIR bands without filter aliases)
        nir_bands = {'J', 'H', 'K'}
        for site in extinction:
            for band in extinction[site]:
                assert band in canonical_bands or band in nir_bands, (
                    f"Extinction band '{band}' at site {site!r} not in filterst canonical bands or NIR bands"
                )

    def test_all_filterst_canonical_bands_in_extinction(self):
        """Every canonical band in filterst should have extinction defined."""
        from lsc.sites import extinction, filterst
        # Exclude JHK-only bands that might not have extinction
        for site in extinction:
            for band in 'UBVRIugriz':
                assert band in extinction[site]
