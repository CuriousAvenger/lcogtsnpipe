DATASETS="l101 sn2017cbv"
TARGET=SN2017cbv
EPOCH=20170515
STANDARD_CATALOG=L101_landolt_stetson.cat
LOCAL_CATALOG=SN2017cbv_landolt_20260115_01_02.cat

run_case() {
    run_pipeline "ingesttar.py -f /datasets/$(dataset_archive l101)"
    run_pipeline "LCOGTingest.py -n SN2017cbv -s 2017-05-15 -e 2017-05-19 -t EXPOSE -r reduced --public"
    run_pipeline 'comparecatalogs.py'
    run_pipeline 'mysql -h db -D supernova -u supernova -psupernova --skip-ssl -e "UPDATE targets SET classificationid=1 WHERE id=1"'
    run_pipeline "mysql -h db -D supernova -u supernova -psupernova --skip-ssl -e \"UPDATE targets SET landolt_cat='$STANDARD_CATALOG' WHERE id=1\""

    run_pipeline "cp /lcogtsnpipe/trunk/src/lsc/standard/cat/landolt/$STANDARD_CATALOG /supernova/standard/cat/landolt/"
    run_pipeline "lscloop.py -n $TARGET -e $EPOCH -f landolt --standard all"
    run_pipeline "lscloop.py -n $TARGET -e $EPOCH -f landolt -s psf"
    run_pipeline "lscloop.py -n $TARGET -e $EPOCH -f landolt -s psfmag"
    run_pipeline "lscloop.py -n $TARGET -e $EPOCH -f landolt -s zcat"
    run_pipeline "lscloop.py -n $TARGET -e $EPOCH -f landolt --standard all -s psf"
    run_pipeline "lscloop.py -n $TARGET -e $EPOCH -f landolt --standard all -s psf -f U --fwhm 4.5 --datamax 5000"
    run_pipeline "lscloop.py -n $TARGET -e $EPOCH -f landolt --standard all -s zcat --field landolt"
    run_pipeline "lscloop.py -n $TARGET -e $EPOCH -f landolt --standard all -s local"

    run_pipeline "cp $LOCAL_CATALOG /supernova/standard/cat/landolt/"
    run_pipeline "mysql -h db -D supernova -u supernova -psupernova --skip-ssl -e \"UPDATE targets SET landolt_cat='$LOCAL_CATALOG' WHERE id=1\""
    run_pipeline "lscloop.py -n $TARGET -e $EPOCH -f landolt -s zcat -F"
    run_pipeline "lscloop.py -n $TARGET -e $EPOCH -f landolt -s mag -F"
    run_pipeline "python /lcogtsnpipe/ci/reduction/compare.py uband --target '$TARGET' --epoch '$EPOCH' --standard-id 1 --standard-catalog '$STANDARD_CATALOG' --local-catalog '$LOCAL_CATALOG'"
}
