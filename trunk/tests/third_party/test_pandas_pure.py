"""
Tests for pandas usage in lcogtsnpipe.

pandas is installed in the lcogtsnpipe environment and is available for
pipeline data analysis tasks (photometry tables, light curves, catalog
manipulation). While not heavily imported in the current source, it is a
declared dependency and its core functionality is exercised here to catch
version-incompatibility regressions.

Tests cover the patterns most relevant to an astronomical photometry pipeline:
- DataFrame creation from catalogs
- Filtering / boolean indexing
- Group-by aggregation (per-filter statistics)
- Reading / writing CSV photometry files
- Time-series indexing for light curves
- Merging / joining catalogs

Coverage additions:
- sort_values by MJD for light-curve ordering
- reset_index / set_index for time-series access
- fillna / dropna on DataFrames
- rename columns (mirror astropy rename_column pattern)
- unique / value_counts — filter & instrument inventory
- apply for flux-to-magnitude row transforms
- pivot_table — multi-filter photometry grids
- to_dict / from_records — dict→DataFrame pipeline pattern
- DataFrame.assign — method chaining
- isnull / notnull column masks
- duplicated / drop_duplicates — catalog deduplication
- between — range filtering
- str accessor — filename/filter string operations
- pd.to_datetime — MJD/ISO date parsing
- DataFrame.copy — safe mutation guard
- Series.describe — summary statistics
"""
import numpy as np
import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

class TestPandasImport:
    def test_pandas_importable(self):
        import pandas as pd
        assert hasattr(pd, "__version__")

    def test_dataframe_importable(self):
        from pandas import DataFrame
        assert callable(DataFrame)

    def test_series_importable(self):
        from pandas import Series
        assert callable(Series)


# ---------------------------------------------------------------------------
# DataFrame creation and basic access
# ---------------------------------------------------------------------------

class TestDataFrameCreation:
    def test_from_dict(self):
        import pandas as pd
        df = pd.DataFrame({
            "filename": ["a.fits", "b.fits"],
            "filter": ["r", "r"],
            "mag": [18.5, 19.0],
            "magerr": [0.02, 0.03],
        })
        assert len(df) == 2
        assert list(df.columns) == ["filename", "filter", "mag", "magerr"]

    def test_from_numpy_structured_array(self):
        import pandas as pd
        dt = np.dtype([("mjd", float), ("mag", float), ("magerr", float)])
        arr = np.array([(58970.5, 18.5, 0.03), (58971.0, 18.7, 0.04)], dtype=dt)
        df = pd.DataFrame(arr)
        assert len(df) == 2
        assert "mjd" in df.columns

    def test_column_access(self):
        import pandas as pd
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        np.testing.assert_array_equal(df["a"].values, [1, 2, 3])

    def test_iloc_row_access(self):
        import pandas as pd
        df = pd.DataFrame({"x": [10, 20, 30]})
        assert df.iloc[1]["x"] == 20

    def test_add_column(self):
        import pandas as pd
        df = pd.DataFrame({"mag": [15.0, 17.0, 19.0]})
        df["flux"] = 10 ** (-(df["mag"] - 25.0) / 2.5)
        assert "flux" in df.columns
        assert all(df["flux"] > 0)


# ---------------------------------------------------------------------------
# Boolean filtering — mirrors pipeline catalog cuts
# ---------------------------------------------------------------------------

class TestDataFrameFiltering:
    def _make_phot(self, n=20):
        import pandas as pd
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "filter": rng.choice(["r", "g", "i"], n),
            "mag": rng.uniform(16, 22, n),
            "magerr": rng.uniform(0.01, 0.2, n),
            "quality": rng.choice([0, 1], n),
        })

    def test_filter_by_quality_flag(self):
        import pandas as pd
        df = self._make_phot()
        good = df[df["quality"] == 0]
        assert len(good) < len(df) or len(good) == len(df)
        assert all(good["quality"] == 0)

    def test_filter_by_magnitude_limit(self):
        import pandas as pd
        df = self._make_phot(30)
        bright = df[df["mag"] < 19.0]
        assert all(bright["mag"] < 19.0)

    def test_filter_by_band(self):
        import pandas as pd
        df = self._make_phot(30)
        rband = df[df["filter"] == "r"]
        assert all(rband["filter"] == "r")

    def test_compound_filter(self):
        import pandas as pd
        df = self._make_phot(50)
        good_r = df[(df["filter"] == "r") & (df["magerr"] < 0.1)]
        assert all(good_r["filter"] == "r")
        assert all(good_r["magerr"] < 0.1)

    def test_notna_filter(self):
        import pandas as pd
        df = pd.DataFrame({"mag": [15.0, None, 18.0, None, 20.0]})
        clean = df[df["mag"].notna()]
        assert len(clean) == 3


# ---------------------------------------------------------------------------
# GroupBy aggregation — per-filter photometry statistics
# ---------------------------------------------------------------------------

class TestGroupBy:
    def test_groupby_mean_mag(self):
        import pandas as pd
        df = pd.DataFrame({
            "filter": ["r", "r", "g", "g"],
            "mag": [18.0, 18.5, 17.0, 17.5],
        })
        means = df.groupby("filter")["mag"].mean()
        assert means["r"] == pytest.approx(18.25)
        assert means["g"] == pytest.approx(17.25)

    def test_groupby_count(self):
        import pandas as pd
        df = pd.DataFrame({"filter": ["r", "g", "r", "r", "i"]})
        counts = df.groupby("filter").size()
        assert counts["r"] == 3
        assert counts["g"] == 1

    def test_groupby_std(self):
        import pandas as pd
        rng = np.random.default_rng(42)
        mags = rng.normal(18.5, 0.1, 20)
        df = pd.DataFrame({"filter": ["r"] * 20, "mag": mags})
        std = df.groupby("filter")["mag"].std()
        assert std["r"] == pytest.approx(np.std(mags, ddof=1), rel=0.01)

    def test_groupby_agg_multiple(self):
        import pandas as pd
        df = pd.DataFrame({
            "filter": ["r", "r", "g"],
            "mag": [18.0, 18.5, 17.0],
            "magerr": [0.02, 0.03, 0.01],
        })
        agg = df.groupby("filter").agg({"mag": "mean", "magerr": "mean"})
        assert "mag" in agg.columns


# ---------------------------------------------------------------------------
# CSV I/O — photometry file read/write
# ---------------------------------------------------------------------------

class TestCSVIO:
    def test_write_read_roundtrip(self, tmp_path):
        import pandas as pd
        df = pd.DataFrame({
            "mjd": [58970.5, 58971.0, 58972.3],
            "mag": [18.5, 18.7, 18.6],
            "magerr": [0.03, 0.04, 0.03],
            "filter": ["r", "r", "r"],
        })
        path = tmp_path / "phot.csv"
        df.to_csv(str(path), index=False)
        df2 = pd.read_csv(str(path))
        assert len(df2) == 3
        np.testing.assert_allclose(df2["mag"].values, [18.5, 18.7, 18.6])

    def test_read_csv_with_comment(self, tmp_path):
        import pandas as pd
        path = tmp_path / "data.csv"
        path.write_text("# header comment\nmjd,mag\n58970.5,18.5\n")
        df = pd.read_csv(str(path), comment="#")
        assert len(df) == 1

    def test_write_tsv(self, tmp_path):
        import pandas as pd
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = tmp_path / "data.tsv"
        df.to_csv(str(path), sep="\t", index=False)
        df2 = pd.read_csv(str(path), sep="\t")
        np.testing.assert_array_equal(df2["a"].values, [1, 2])


# ---------------------------------------------------------------------------
# Merge / join — catalog cross-matching result
# ---------------------------------------------------------------------------

class TestMerge:
    def test_inner_join_on_filename(self):
        import pandas as pd
        phot = pd.DataFrame({
            "filename": ["a.fits", "b.fits", "c.fits"],
            "mag": [18.0, 18.5, 19.0],
        })
        meta = pd.DataFrame({
            "filename": ["a.fits", "c.fits"],
            "airmass": [1.2, 1.5],
        })
        merged = pd.merge(phot, meta, on="filename", how="inner")
        assert len(merged) == 2
        assert "airmass" in merged.columns

    def test_left_join_preserves_all_left(self):
        import pandas as pd
        left = pd.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
        right = pd.DataFrame({"id": [1, 3], "extra": [100, 300]})
        merged = pd.merge(left, right, on="id", how="left")
        assert len(merged) == 3
        assert merged.loc[merged["id"] == 2, "extra"].isna().all()

    def test_concat_rows(self):
        import pandas as pd
        df1 = pd.DataFrame({"mjd": [1.0, 2.0], "mag": [18.0, 18.5]})
        df2 = pd.DataFrame({"mjd": [3.0, 4.0], "mag": [19.0, 19.5]})
        combined = pd.concat([df1, df2], ignore_index=True)
        assert len(combined) == 4
        assert combined.iloc[3]["mag"] == 19.5


# ---------------------------------------------------------------------------
# Series statistics — common in pipeline summary reports
# ---------------------------------------------------------------------------

class TestSeriesStatistics:
    def test_series_mean(self):
        import pandas as pd
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert s.mean() == pytest.approx(3.0)

    def test_series_median(self):
        import pandas as pd
        s = pd.Series([1.0, 2.0, 100.0])
        assert s.median() == pytest.approx(2.0)

    def test_series_dropna(self):
        import pandas as pd
        s = pd.Series([1.0, None, 3.0, None, 5.0])
        clean = s.dropna()
        assert len(clean) == 3

    def test_series_clip(self):
        import pandas as pd
        s = pd.Series([-100.0, 0.0, 50.0, 200.0])
        clipped = s.clip(0, 100)
        assert clipped.iloc[0] == 0.0
        assert clipped.iloc[-1] == 100.0


# ---------------------------------------------------------------------------
# Sort — ordering light curves by time
# ---------------------------------------------------------------------------

class TestSortValues:
    """sort_values used to ensure time-ordered light-curve output."""

    def test_sort_by_mjd_ascending(self):
        """Pipeline light curves must be time-ordered before plotting."""
        import pandas as pd
        df = pd.DataFrame({"mjd": [58972.0, 58970.5, 58971.3], "mag": [18.6, 18.5, 18.7]})
        sorted_df = df.sort_values("mjd").reset_index(drop=True)
        assert sorted_df["mjd"].iloc[0] == pytest.approx(58970.5)
        assert sorted_df["mjd"].iloc[-1] == pytest.approx(58972.0)

    def test_sort_by_multiple_columns(self):
        """Sort by filter then mjd — groups each band's light curve."""
        import pandas as pd
        df = pd.DataFrame({
            "filter": ["r", "g", "r", "g"],
            "mjd": [58972.0, 58970.0, 58970.5, 58971.0],
            "mag": [18.6, 17.0, 18.5, 17.1],
        })
        sorted_df = df.sort_values(["filter", "mjd"]).reset_index(drop=True)
        # g band should come first (alphabetical)
        assert sorted_df["filter"].iloc[0] == "g"

    def test_sort_descending(self):
        import pandas as pd
        df = pd.DataFrame({"mag": [19.0, 17.0, 18.0]})
        desc = df.sort_values("mag", ascending=False).reset_index(drop=True)
        assert desc["mag"].iloc[0] == pytest.approx(19.0)


# ---------------------------------------------------------------------------
# Index management — set_index / reset_index for time-series lookups
# ---------------------------------------------------------------------------

class TestIndexManagement:
    """set_index / reset_index patterns for time-series photometry access."""

    def test_set_index_on_filename(self):
        import pandas as pd
        df = pd.DataFrame({
            "filename": ["a.fits", "b.fits", "c.fits"],
            "mag": [18.0, 18.5, 19.0],
        })
        indexed = df.set_index("filename")
        assert indexed.loc["b.fits", "mag"] == pytest.approx(18.5)

    def test_reset_index_restores_integer_index(self):
        import pandas as pd
        df = pd.DataFrame({"mjd": [1.0, 2.0, 3.0], "mag": [18.0, 18.5, 19.0]})
        df2 = df.set_index("mjd").reset_index()
        assert "mjd" in df2.columns
        assert list(df2.index) == [0, 1, 2]

    def test_sort_index_after_set(self):
        import pandas as pd
        df = pd.DataFrame({"mjd": [3.0, 1.0, 2.0], "mag": [19.0, 18.0, 18.5]})
        df = df.set_index("mjd").sort_index()
        assert df.index[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# fillna / dropna on DataFrames — handling missing pipeline data
# ---------------------------------------------------------------------------

class TestMissingData:
    """fillna and dropna used to handle incomplete photometry records."""

    def test_dropna_removes_incomplete_rows(self):
        import pandas as pd
        df = pd.DataFrame({
            "mjd": [1.0, 2.0, 3.0],
            "mag": [18.0, None, 19.0],
            "magerr": [0.02, 0.03, None],
        })
        clean = df.dropna()
        assert len(clean) == 1  # only row 0 has both values

    def test_dropna_subset_column(self):
        import pandas as pd
        df = pd.DataFrame({"mag": [18.0, None, 19.0], "magerr": [0.02, 0.03, None]})
        clean = df.dropna(subset=["mag"])
        assert len(clean) == 2

    def test_fillna_with_sentinel(self):
        """Pipeline uses 9999 as a fill value for missing magnitudes."""
        import pandas as pd
        df = pd.DataFrame({"mag": [18.0, None, 19.0]})
        filled = df["mag"].fillna(9999.0)
        assert filled.iloc[1] == pytest.approx(9999.0)

    def test_fillna_forward_fill(self):
        import pandas as pd
        s = pd.Series([1.0, None, None, 4.0])
        filled = s.ffill()
        assert filled.iloc[1] == pytest.approx(1.0)
        assert filled.iloc[2] == pytest.approx(1.0)

    def test_isnull_notnull(self):
        import pandas as pd
        df = pd.DataFrame({"mag": [18.0, None, 19.0]})
        assert df["mag"].isnull().sum() == 1
        assert df["mag"].notnull().sum() == 2


# ---------------------------------------------------------------------------
# rename — mirrors astropy Table.rename_column used in lscabsphotdef
# ---------------------------------------------------------------------------

class TestRename:
    """DataFrame.rename mirrors astropy rename_column in the pipeline."""

    def test_rename_single_column(self):
        """lscabsphotdef: t.rename_column('ymag', 'umag')"""
        import pandas as pd
        df = pd.DataFrame({"ymag": [20.0, 21.0], "e_ymag": [0.05, 0.06]})
        df = df.rename(columns={"ymag": "umag", "e_ymag": "e_umag"})
        assert "umag" in df.columns
        assert "e_umag" in df.columns
        assert "ymag" not in df.columns

    def test_rename_with_function(self):
        import pandas as pd
        df = pd.DataFrame({"A": [1], "B": [2]})
        df = df.rename(columns=str.lower)
        assert list(df.columns) == ["a", "b"]

    def test_rename_inplace(self):
        import pandas as pd
        df = pd.DataFrame({"old": [1, 2, 3]})
        df.rename(columns={"old": "new"}, inplace=True)
        assert "new" in df.columns


# ---------------------------------------------------------------------------
# unique / value_counts — filter inventory and deduplication
# ---------------------------------------------------------------------------

class TestUniqueAndValueCounts:
    """Inventory of filters/instruments observed in a run."""

    def test_unique_filters(self):
        import pandas as pd
        df = pd.DataFrame({"filter": ["r", "r", "g", "i", "g", "r"]})
        filters = sorted(df["filter"].unique())
        assert filters == ["g", "i", "r"]

    def test_value_counts_descending(self):
        import pandas as pd
        df = pd.DataFrame({"filter": ["r", "r", "r", "g", "g", "i"]})
        vc = df["filter"].value_counts()
        assert vc.index[0] == "r"
        assert vc["r"] == 3

    def test_nunique_count(self):
        import pandas as pd
        df = pd.DataFrame({"instrument": ["fa01", "fa02", "fa01", "fa03"]})
        assert df["instrument"].nunique() == 3


# ---------------------------------------------------------------------------
# apply — per-row flux/magnitude transformations
# ---------------------------------------------------------------------------

class TestApply:
    """DataFrame.apply for row-wise photometry conversions."""

    def test_apply_flux_to_mag(self):
        """Convert flux column to magnitude: m = -2.5 * log10(flux) + zp"""
        import pandas as pd
        zp = 25.0
        df = pd.DataFrame({"flux": [100.0, 10.0, 1.0]})
        df["mag"] = df["flux"].apply(lambda f: -2.5 * np.log10(f) + zp)
        assert df["mag"].iloc[0] == pytest.approx(20.0)
        assert df["mag"].iloc[1] == pytest.approx(22.5)
        assert df["mag"].iloc[2] == pytest.approx(25.0)

    def test_apply_row_wise(self):
        import pandas as pd
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        df["hyp"] = df.apply(lambda row: np.sqrt(row["a"] ** 2 + row["b"] ** 2), axis=1)
        assert df["hyp"].iloc[0] == pytest.approx(np.sqrt(10.0))

    def test_apply_string_transform(self):
        import pandas as pd
        df = pd.DataFrame({"filename": ["IMG_001.fits", "IMG_002.fits"]})
        df["stem"] = df["filename"].apply(lambda f: f.replace(".fits", ""))
        assert df["stem"].iloc[0] == "IMG_001"


# ---------------------------------------------------------------------------
# pivot_table — multi-filter photometry grid
# ---------------------------------------------------------------------------

class TestPivotTable:
    """pivot_table for cross-filter magnitude comparisons."""

    def test_pivot_mean_mag_by_filter_and_night(self):
        import pandas as pd
        df = pd.DataFrame({
            "night": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "filter": ["r", "g", "r", "g"],
            "mag": [18.0, 17.0, 18.5, 17.5],
        })
        pt = df.pivot_table(values="mag", index="night", columns="filter", aggfunc="mean")
        assert pt.loc["2024-01-01", "r"] == pytest.approx(18.0)
        assert pt.loc["2024-01-02", "g"] == pytest.approx(17.5)

    def test_pivot_count_observations(self):
        import pandas as pd
        df = pd.DataFrame({
            "filter": ["r", "r", "g", "g", "g"],
            "night": ["n1", "n2", "n1", "n2", "n2"],
        })
        pt = df.pivot_table(index="night", columns="filter", aggfunc="size", fill_value=0)
        assert pt.loc["n2", "g"] == 2


# ---------------------------------------------------------------------------
# to_dict / from_records — dict↔DataFrame pipeline conversion
# ---------------------------------------------------------------------------

class TestDictConversion:
    """Mirrors pipeline pattern where DB query results (dicts) become DataFrames."""

    def test_from_records(self):
        import pandas as pd
        records = [
            {"filename": "a.fits", "mag": 18.0, "filter": "r"},
            {"filename": "b.fits", "mag": 18.5, "filter": "r"},
        ]
        df = pd.DataFrame.from_records(records)
        assert len(df) == 2
        assert "filename" in df.columns

    def test_to_dict_records(self):
        import pandas as pd
        df = pd.DataFrame({"filename": ["a.fits"], "mag": [18.0]})
        records = df.to_dict(orient="records")
        assert isinstance(records, list)
        assert records[0]["filename"] == "a.fits"

    def test_to_dict_list(self):
        import pandas as pd
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        d = df.to_dict(orient="list")
        assert d["a"] == [1, 2]
        assert d["b"] == [3, 4]


# ---------------------------------------------------------------------------
# assign — method-chaining column additions
# ---------------------------------------------------------------------------

class TestAssign:
    """DataFrame.assign for non-mutating column additions."""

    def test_assign_new_column(self):
        import pandas as pd
        df = pd.DataFrame({"mag": [18.0, 19.0, 20.0]})
        df2 = df.assign(flux=lambda x: 10 ** (-(x["mag"] - 25.0) / 2.5))
        assert "flux" in df2.columns
        assert "flux" not in df.columns  # original unchanged

    def test_assign_derived_column(self):
        import pandas as pd
        df = pd.DataFrame({"mag": [18.0], "magerr": [0.05]})
        df2 = df.assign(snr=lambda x: 1.0 / x["magerr"])
        assert df2["snr"].iloc[0] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# drop_duplicates — catalog deduplication
# ---------------------------------------------------------------------------

class TestDropDuplicates:
    """Catalog cross-matches can produce duplicate rows; these must be removed."""

    def test_drop_duplicate_filenames(self):
        import pandas as pd
        df = pd.DataFrame({
            "filename": ["a.fits", "b.fits", "a.fits"],
            "mag": [18.0, 18.5, 18.0],
        })
        dedup = df.drop_duplicates(subset=["filename"])
        assert len(dedup) == 2

    def test_keep_last_duplicate(self):
        import pandas as pd
        df = pd.DataFrame({
            "id": [1, 1, 2],
            "mag": [18.0, 18.1, 19.0],
        })
        dedup = df.drop_duplicates(subset=["id"], keep="last").reset_index(drop=True)
        assert dedup.loc[dedup["id"] == 1, "mag"].iloc[0] == pytest.approx(18.1)

    def test_duplicated_mask(self):
        import pandas as pd
        df = pd.DataFrame({"filter": ["r", "r", "g"]})
        mask = df.duplicated(subset=["filter"])
        assert mask.sum() == 1  # second "r" is a duplicate


# ---------------------------------------------------------------------------
# between — range filtering for magnitude/time windows
# ---------------------------------------------------------------------------

class TestBetween:
    """Series.between for pipeline magnitude/time windowing."""

    def test_between_magnitude_range(self):
        import pandas as pd
        s = pd.Series([14.0, 17.0, 19.5, 22.0, 25.0])
        in_range = s.between(16.0, 21.0)
        assert in_range.sum() == 2

    def test_between_inclusive_endpoints(self):
        import pandas as pd
        s = pd.Series([15.0, 16.0, 17.0, 18.0])
        mask = s.between(16.0, 17.0)
        assert mask.sum() == 2  # 16.0 and 17.0 included

    def test_between_time_window(self):
        import pandas as pd
        mjds = pd.Series([58970.0, 58971.5, 58975.0, 58980.0])
        in_window = mjds.between(58970.0, 58976.0)
        assert in_window.sum() == 3


# ---------------------------------------------------------------------------
# str accessor — filename/filter string operations
# ---------------------------------------------------------------------------

class TestStrAccessor:
    """Series.str accessor for string column manipulation."""

    def test_str_contains_filter(self):
        import pandas as pd
        df = pd.DataFrame({"filename": ["coj1m003-fa12-20240101-0001-e91.fits",
                                         "ogg2m001-ep05-20240102-0002-e91.fits"]})
        coj = df[df["filename"].str.contains("coj")]
        assert len(coj) == 1

    def test_str_replace_extension(self):
        import pandas as pd
        s = pd.Series(["a.fits", "b.fits", "c.fits"])
        stem = s.str.replace(".fits", "", regex=False)
        assert stem.iloc[0] == "a"

    def test_str_split_and_get(self):
        import pandas as pd
        s = pd.Series(["coj1m003-fa12-20240101-0001-e91.fits"])
        site = s.str.split("-").str[0]
        assert site.iloc[0] == "coj1m003"

    def test_str_upper_lower(self):
        import pandas as pd
        s = pd.Series(["R", "G", "I"])
        lower = s.str.lower()
        assert list(lower) == ["r", "g", "i"]


# ---------------------------------------------------------------------------
# pd.to_datetime — date/time parsing from FITS headers
# ---------------------------------------------------------------------------

class TestDatetimeParsing:
    """pd.to_datetime converts ISO date-obs strings to Timestamp objects."""

    def test_parse_iso_dateobs(self):
        import pandas as pd
        dates = pd.to_datetime(["2024-01-15T03:22:00", "2024-01-16T04:10:00"])
        assert dates[0].year == 2024
        assert dates[0].month == 1
        assert dates[0].day == 15

    def test_datetime_difference_in_days(self):
        import pandas as pd
        t1 = pd.Timestamp("2024-01-01")
        t2 = pd.Timestamp("2024-01-11")
        delta = (t2 - t1).days
        assert delta == 10

    def test_datetime_series_sort(self):
        import pandas as pd
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-15", "2024-01-10", "2024-01-20"]),
            "mag": [18.5, 18.0, 19.0],
        })
        df_sorted = df.sort_values("date").reset_index(drop=True)
        assert df_sorted["date"].iloc[0] == pd.Timestamp("2024-01-10")


# ---------------------------------------------------------------------------
# DataFrame.copy — safe mutation guard
# ---------------------------------------------------------------------------

class TestDataFrameCopy:
    """copy() prevents accidental mutation of the original DataFrame."""

    def test_copy_is_independent(self):
        import pandas as pd
        df = pd.DataFrame({"mag": [18.0, 19.0, 20.0]})
        df_copy = df.copy()
        df_copy["mag"] = 99.0
        assert df["mag"].iloc[0] == pytest.approx(18.0)  # original unchanged

    def test_copy_has_same_values(self):
        import pandas as pd
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = df.copy()
        np.testing.assert_array_equal(df["a"].values, df2["a"].values)


# ---------------------------------------------------------------------------
# Series.describe — summary statistics for pipeline QA
# ---------------------------------------------------------------------------

class TestDescribe:
    """Series.describe used in pipeline QA reports."""

    def test_describe_count_mean_std(self):
        import pandas as pd
        rng = np.random.default_rng(99)
        mags = rng.normal(18.5, 0.2, 50)
        s = pd.Series(mags)
        desc = s.describe()
        assert desc["count"] == 50
        assert desc["mean"] == pytest.approx(np.mean(mags))
        assert desc["std"] == pytest.approx(np.std(mags, ddof=1), rel=1e-5)

    def test_describe_percentiles(self):
        import pandas as pd
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        desc = s.describe()
        assert desc["50%"] == pytest.approx(3.0)
        assert desc["min"] == pytest.approx(1.0)
        assert desc["max"] == pytest.approx(5.0)
