DATASETS="sn2018cow"
TARGET=SN2018cow
EPOCH=20180701

run_case() {
    run_pipeline "ingesttar.py -f /datasets/$(dataset_archive sn2018cow)"
    run_pipeline 'comparecatalogs.py'
    run_pipeline 'lscloop.py -n SN2018cow -e 20180701-20190901'
    run_pipeline 'lscloop.py -n 2018cow -e 20190831 -s psf'
    run_pipeline 'lscloop.py -n 2018cow -e 20190831 --id 87 -s template'
    run_pipeline 'lscloop.py -n 2018cow -e 20190831 --id 89 -s template'
    run_pipeline 'lscloop.py -n 2018cow -e 20190831 --id 91 -s template'
    run_pipeline 'lscloop.py -n SN2018cow -e 20190831 --filetype 4 -s cosmic'
    run_pipeline 'lscloop.py -n SN2018cow -e 20190831 --filetype 4 -s psf --nstars 12 --datamax 6000 --fwhm 4'
    run_pipeline 'lscloop.py -n SN2018cow -e 20180701 -s cosmic'
    run_pipeline 'lscloop.py -n SN2018cow -e 20180701 -s psf --nstars 12 --datamax 60000'
    run_pipeline 'lscloop.py -n SN2018cow -e 20180701 --normalize t -T fl --tempdate 20190831 --temptel fa --difftype 0 --fixpix -s diff'
    run_pipeline 'lscloop.py -n SN2018cow -e 20180701 --filetype 3 --difftype 0 -s psf'
    run_pipeline 'lscloop.py -n SN2018cow -e 20180701 --filetype 3 -s psfmag -x 1 -y 1'
    run_pipeline 'lscloop.py -n SN2018cow -e 20180701 --filetype 3 -f sloan -s zcat --type fit'
    run_pipeline 'lscloop.py -n SN2018cow -e 20180701 --filetype 3 -s mag --type fit'
    run_pipeline 'python /lcogtsnpipe/ci/reduction/compare.py difference --target SN2018cow --science-epoch 20180701 --template-epoch 20190831 --expected-science 6 --expected-templates 3'
}
