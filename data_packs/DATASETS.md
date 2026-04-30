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

## Recommended Real-World Packs

- `ohio_t1dm`: classic benchmark for CGM + insulin + meals + daily-life events. Best when you want controller or prediction comparisons against a well-known baseline.
- `diatrend`: larger controlled-access CGM + pump dataset with carb logs and pump settings. Best for algorithm development once you want more subject diversity than OhioT1DM.
- `t1d_uom`: 12-week multimodal dataset with CGM, basal/bolus insulin, detailed meal macros, activity, and sleep. Best for research that links glucose to lifestyle and recovery.
- `t1d_granada`: very large longitudinal glucose-focused dataset from 736 people with T1D. Best for population-scale glucose pattern work, but not ideal for controller evaluation because it is not meal/insulin complete.
- `azt1d` and `hupa_ucm`: already in the registry and still strong choices when you need meal + insulin traces that are easier to work with than request-gated datasets.
