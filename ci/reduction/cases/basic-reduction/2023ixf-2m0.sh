run_case() {
    run_pipeline 'LCOGTingest.py -n 2023ixf -s 2026-01-01 -e 2026-05-01 -t EXPOSE -r reduced -T 2m0a'
    run_pipeline 'comparecatalogs.py'
    run_pipeline 'lscloop.py -n 2023ixf -e 2026-2027 -T 2m0'
    run_pipeline 'lscloop.py -n 2023ixf -e 2026-2027 -T 2m0 -s wcs'
    run_pipeline 'lscloop.py -n 2023ixf -e 2026-2027 -T 2m0 -s psf'
    run_pipeline 'lscloop.py -n 2023ixf -e 2026-2027 -T 2m0 -s psfmag'
    run_pipeline 'lscloop.py -n 2023ixf -e 2026-2027 -T 2m0 -s zcat --field apass'
    run_pipeline 'lscloop.py -n 2023ixf -e 2026-2027 -T 2m0 -s mag'
}
