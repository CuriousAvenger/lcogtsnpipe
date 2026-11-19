run_case() {
    run_pipeline "LCOGTingest.py -n 2017cbv -s 2017-05-15 -e 2017-05-19 -t EXPOSE -r reduced --public"
    run_pipeline "comparecatalogs.py"
    run_pipeline "lscloop.py -n 2017cbv -e 2017-2018"
    run_pipeline "lscloop.py -n 2017cbv -e 2017-2018 -s psf"
    run_pipeline "lscloop.py -n 2017cbv -e 2017-2018 -s psfmag"
    run_pipeline "lscloop.py -n 2017cbv -e 2017-2018 -s zcat"
    run_pipeline "lscloop.py -n 2017cbv -e 2017-2018 -s mag"
}
