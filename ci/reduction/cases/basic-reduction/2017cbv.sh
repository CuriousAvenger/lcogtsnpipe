DATASETS="sn2017cbv"
TARGET=SN2017cbv
EPOCH=20170515
EXPECTED_IMAGES=12
EXPECTED_CALIBRATED_IMAGES=10

run_case() {
    run_pipeline "ingesttar.py -f /datasets/$(dataset_archive sn2017cbv)"
    run_pipeline "comparecatalogs.py"
    run_pipeline "lscloop.py -n $TARGET -e $EPOCH"
    run_pipeline "lscloop.py -n $TARGET -e $EPOCH -s psf --field apass --nstars 12 --datamax 75000"
    run_pipeline "lscloop.py -n $TARGET -e $EPOCH -s psfmag"
    run_pipeline "lscloop.py -n $TARGET -e $EPOCH -s zcat"
    run_pipeline "lscloop.py -n $TARGET -e $EPOCH -s mag"
    run_pipeline "python /lcogtsnpipe/ci/reduction/compare.py basic --target '$TARGET' --epoch '$EPOCH' --expected-images '$EXPECTED_IMAGES' --expected-calibrated-images '$EXPECTED_CALIBRATED_IMAGES'"
}
