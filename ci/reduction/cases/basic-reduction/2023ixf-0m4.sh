run_case() {
    run_pipeline 'LCOGTingest.py -n 2023ixf -s 2023-05-20 -e 2023-05-21 -t EXPOSE -r reduced -T 0m4b'
    run_pipeline 'comparecatalogs.py'
    run_pipeline 'lscloop.py -n 2023ixf -e 2023-2024 -T 0m4'
    run_pipeline 'lscloop.py -n 2023ixf -e 2023-2024 -T 0m4 -s psf'
    run_pipeline 'lscloop.py -n 2023ixf -e 2023-2024 -T 0m4 -s psfmag'
    run_pipeline 'lscloop.py -n 2023ixf -e 2023-2024 -T 0m4 -s zcat'
    run_pipeline 'lscloop.py -n 2023ixf -e 2023-2024 -T 0m4 -s mag'
}
