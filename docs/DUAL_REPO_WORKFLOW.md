# Bundled MDMP Workflow

The SDK no longer depends on a separate public MDMP repository checkout.

## What changed

MDMP now ships inside the SDK source tree as bundled Python packages:

- `src/mdmp_core/`
- `src/mdmp_ai/`
- `src/mdmp_flavors/`
- `src/mdmp_integrations/`

That means:

- the SDK can keep its `iints mdmp ...` namespace
- local AI MDMP certification can keep using `mdmp_core`
- the standalone `mdmp` CLI can still be exposed from the SDK package
- the old dual-repo maintenance flow is no longer required

## CI sync gate

The compatibility gate remains, but it now checks:

- SDK wrapper behavior
- bundled `mdmp_core` behavior

Workflow:

- `.github/workflows/mdmp-sync.yml`

Check script:

- `tools/ci/check_mdmp_sync.py`

## Practical install modes

### SDK only

```bash
python -m pip install -U "iints-sdk-python35"
```

### SDK with bundled MDMP crypto/signing support

```bash
python -m pip install -U "iints-sdk-python35[mdmp]"
```

The `mdmp` extra now enables the cryptographic pieces needed for signing and verification.

## MDMP backend in SDK

SDK MDMP commands can still use either backend name:

- `iints` (built-in contract/runtime path)
- `mdmp_core` (bundled MDMP package path)

Set backend explicitly if you want the bundled `mdmp_core` route:

```bash
export IINTS_MDMP_BACKEND=mdmp_core
```

Then run:

```bash
iints mdmp validate mdmp_contract.yaml data/my_cgm.csv --output-json results/mdmp_report.json
```

The command summary prints the active backend so provenance stays explicit.
