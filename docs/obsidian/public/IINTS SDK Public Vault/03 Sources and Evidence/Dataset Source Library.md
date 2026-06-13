---
tags:
  - iints/sources
  - iints/data
  - iints/user
cssclasses:
  - iints-dashboard
status: active
updated: 2026-05-21
---
# Dataset Source Library

This page is written for an SDK user who wants to know **which datasets exist, whether they are bundled, public, manual, or request-only, and how they support realism/training workflows.**

| Dataset ID | Name | Access | License / terms | Landing page | DOI | Note |
| --- | --- | --- | --- | --- | --- | --- |
| sample | IINTS Sample CGM (Bundled) | bundled | Demo-only (bundled with SDK) |  |  | [[sample]] |
| ohio_t1dm | OhioT1DM Dataset | request | Research access on request (see dataset page) | https://webpages.charlotte.edu/rbunescu/data/ohiot1dm/OhioT1DM-dataset.html |  | [[ohio_t1dm]] |
| diatrend | DiaTrend Dataset | request | Controlled access via Synapse | https://doi.org/10.7303/syn38187184 |  | [[diatrend]] |
| t1d_uom | T1D-UOM Longitudinal Multimodal Dataset | manual | Open dataset on Zenodo (see record terms) | https://doi.org/10.5281/zenodo.15806142 | 10.5281/zenodo.15806142 | [[t1d_uom]] |
| t1d_granada | T1DiabetesGranada Dataset | request | Zenodo Data Usage Agreement / specific-permission access | https://doi.org/10.5281/zenodo.10050944 | 10.5281/zenodo.10050944 | [[t1d_granada]] |
| aide_t1d | AIDE T1D Public Dataset | public-download | Public dataset (see Jaeb public datasets page) | https://public.jaeb.org/datasets/ |  | [[aide_t1d]] |
| pedap | PEDAP Public Dataset | public-download | Public dataset (see Jaeb public datasets page) | https://public.jaeb.org/datasets/ |  | [[pedap]] |
| azt1d | AZT1D: A Real-World Dataset for Type 1 Diabetes | manual | CC BY 4.0 | https://data.mendeley.com/datasets/gk9m674wcx/1 | 10.17632/gk9m674wcx.1 | [[azt1d]] |
| hupa_ucm | HUPA-UCM Diabetes Dataset | manual | CC BY 4.0 | https://data.mendeley.com/datasets/3hbcscwz44/1 | 10.17632/3hbcscwz44.1 | [[hupa_ucm]] |
| openaps_data_commons | OpenAPS Data Commons | request | Data use agreement (see OpenAPS) | https://openaps.org/outcomes/data-commons/ |  | [[openaps_data_commons]] |
| tidepool_bigdata | Tidepool Big Data Donation | request | Research collaboration / approval required | https://www.tidepool.org/bigdata |  | [[tidepool_bigdata]] |
| niddk_central | NIDDK Central Repository | request | Repository access agreement | https://repository.niddk.nih.gov/ |  | [[niddk_central]] |
| t1d_exchange | T1D Exchange Clinic Registry | request | Data request / approval required | https://datacatalog.med.nyu.edu/dataset/10129 |  | [[t1d_exchange]] |

## User Commands

```bash
iints data list
iints data info ohio_t1dm
iints data cite azt1d
```

## Practical Meaning

- Use **bundled/demo** data for quick demos and offline booth workflows.
- Use **public-download** data for reproducible benchmarks when available.
- Use **manual/request** data only after following the dataset owner's access rules.
- Keep dataset IDs and citations in every study bundle.
