run_case() {
    run_pipeline "LCOGTingest.py -n SN2017cbv -s 2017-05-15 -e 2017-05-19 -t EXPOSE -r reduced --public"
    run_pipeline "comparecatalogs.py"
    run_pipeline "lscloop.py -n $TARGET -e $EPOCH"
    run_pipeline "lscloop.py -n $TARGET -e $EPOCH -s psf --field apass --nstars 12 --datamax 75000"
    run_pipeline "lscloop.py -n $TARGET -e $EPOCH -s psfmag"
    run_pipeline "lscloop.py -n $TARGET -e $EPOCH -s zcat"
    run_pipeline "lscloop.py -n $TARGET -e $EPOCH -s mag"
    run_pipeline "python /lcogtsnpipe/ci/reduction/compare.py basic --target '$TARGET' --epoch '$EPOCH' --expected-images '$EXPECTED_IMAGES' --expected-calibrated-images '$EXPECTED_CALIBRATED_IMAGES'"
}
