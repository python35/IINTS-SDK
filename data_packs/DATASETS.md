# Official Data Sources (Real-World)

This SDK supports **official real-world datasets** via the `iints data` CLI.

## Quick commands

```bash
iints data list
iints data info aide_t1d
iints data fetch aide_t1d
iints data fetch sample --output-dir data_packs/sample
```

## Notes

- Datasets marked **request** require approval from the data owner.
- We do **not** ship full datasets in the repo; download them into `data_packs/official/<dataset>`.
- All datasets are converted into the IINTS universal schema via `iints import-data` or `iints import-wizard`.
- For public downloads, the CLI writes `SHA256SUMS.txt` after download when the source doesn't publish a checksum.
- `iints data info <dataset>` prints BibTeX + citation text.

## Registry

The registry lives in `src/iints/data/datasets.json` and is packaged with the SDK.

If you add a dataset, include:
- Official source link
- Access type (public-download, manual, request)
- License or data use terms
