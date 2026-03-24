from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import typer
import yaml

from mdmp_ai import LineageTracker, load_lineage_card
from mdmp_core.audit import build_audit_from_sources, build_audit_html
from mdmp_core.bias_hooks import run_bias_hooks
from mdmp_core.bundle import build_bundle_from_files, verify_bundle_integrity
from mdmp_core.certification import create_certificate
from mdmp_core.compare import compare_reports
from mdmp_core.conformance import run_conformance_suite, write_conformance_report
from mdmp_core.contracts import load_contract, parse_contract, save_contract
from mdmp_core.crypto import MDMPSigner, MDMPVerifier, generate_keypair, normalize_unsigned_payload
from mdmp_core.delegate import DelegateConstraints, DelegateIssuer, DelegateSigner, DelegateVerifier
from mdmp_core.diffing import compare_datasets
from mdmp_core.drift import compute_drift, format_drift_report, severity_at_or_above, valid_severities
from mdmp_core.fingerprint import check_fingerprint, compute_fingerprint
from mdmp_core.fingerprint_store import FingerprintStore
from mdmp_core.hf import build_hf_mdmp_section, load_report
from mdmp_core.llm_provenance import build_llm_training_card
from mdmp_core.migrate import (
    CURRENT_SPEC_VERSION,
    detect_version,
    iter_json_files,
    load_json,
    migrate_file,
)
from mdmp_core.policy import apply_policy_effects, default_policy, evaluate_policy, load_policy, save_policy
from mdmp_core.prov import card_to_prov
from mdmp_core.registry import (
    export_public_bundle,
    export_signed_public_bundle,
    import_public_bundle,
    init_registry,
    list_records,
    lookup_record,
    sync_public_bundle_from_url,
    upsert_record,
)
from mdmp_core.synthetic import build_synthetic_metadata
from mdmp_core.schema_export import contract_to_frictionless_schema, contract_to_json_schema
from mdmp_core.trust import (
    load_trust_store,
    trust_add_key,
    trust_init as trust_store_init,
    trust_revoke_delegate,
    trust_revoke_key,
    trust_unrevoke_delegate,
    trust_unrevoke_key,
)
from mdmp_core.runner import ContractRunner, dataframe_fingerprint
from mdmp_core.visualizer import build_dashboard_html
from mdmp_integrations.dvc import build_dvc_stage, write_dvc_stage
from mdmp_integrations.mlflow import log_mdmp_artifact_to_mlflow
from mdmp_integrations.wandb import log_mdmp_artifact_to_wandb
from mdmp_flavors import BUILTIN_TEMPLATES, available_flavors, get_template, load_external_templates


app = typer.Typer(help="MDMP CLI - contracts, grading, fingerprints, and lineage")
registry_app = typer.Typer(help="MDMP registry scaffold commands")
flavors_app = typer.Typer(help="Flavor templates (built-in + plugin)")
integrations_app = typer.Typer(help="Integration helpers (MLflow, W&B, DVC)")
authority_app = typer.Typer(help="Authority operations (key generation + signing)")
trust_app = typer.Typer(help="Trust store operations (key rotation and revocation)")
app.add_typer(registry_app, name="registry")
app.add_typer(flavors_app, name="flavors")
app.add_typer(integrations_app, name="integrations")
app.add_typer(authority_app, name="authority")
app.add_typer(trust_app, name="trust")


def safe_output_path(path: Path, base_dir: Path | None = None) -> Path:
    """
    Resolve output paths defensively.

    - Absolute paths are allowed (explicit user intent).
    - Relative paths must stay inside the working directory.
    """
    if path.is_absolute():
        return path
    base = (base_dir or Path.cwd()).resolve()
    resolved = (base / path).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise typer.BadParameter(f"Path traversal detected in output path: {path}") from exc
    return resolved


def prepare_output_path(path: Path, base_dir: Path | None = None) -> Path:
    resolved = safe_output_path(path, base_dir=base_dir)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def resolve_passphrase(
    *,
    passphrase: str | None = None,
    passphrase_env: str | None = None,
    passphrase_file: Path | None = None,
) -> str | None:
    sources = [
        passphrase is not None,
        passphrase_env is not None,
        passphrase_file is not None,
    ]
    if sum(bool(source) for source in sources) > 1:
        raise typer.BadParameter("Use only one of --passphrase, --passphrase-env, or --passphrase-file")

    if passphrase is not None:
        if not passphrase:
            raise typer.BadParameter("Passphrase cannot be empty")
        return passphrase

    if passphrase_env is not None:
        value = os.getenv(passphrase_env)
        if not value:
            raise typer.BadParameter(f"Environment variable is empty or missing: {passphrase_env}")
        return value

    if passphrase_file is not None:
        text = passphrase_file.read_text(encoding="utf-8").strip()
        if not text:
            raise typer.BadParameter(f"Passphrase file is empty: {passphrase_file}")
        return text

    return None


@app.command("init")
def init_contract(
    output: Path = typer.Option(Path("mdmp_contract.yaml"), help="Output contract file"),
    flavor: str = typer.Option("health", help="Template flavor: health|finance|industrial"),
) -> None:
    try:
        payload = get_template(flavor)
    except KeyError as exc:
        raise typer.BadParameter(str(exc))
    contract = parse_contract(payload)
    output = safe_output_path(output)
    save_contract(output, contract)
    typer.echo(f"Created contract: {output}")


@flavors_app.command("list")
def flavors_list() -> None:
    all_flavors = available_flavors()
    built_in = sorted(BUILTIN_TEMPLATES.keys())
    external = load_external_templates()
    payload = {
        "flavors": all_flavors,
        "built_in": built_in,
        "external_plugins": sorted(external.keys()),
    }
    typer.echo(json.dumps(payload, indent=2))


@authority_app.command("keygen")
def authority_keygen(
    output_dir: Path = typer.Option(Path("keys"), help="Output directory for keypair"),
    private_name: str = typer.Option("mdmp_private_v1.pem", help="Private key filename"),
    public_name: str = typer.Option("mdmp_pub_v1.pem", help="Public key filename"),
    encrypt_private_key: bool = typer.Option(
        False,
        "--encrypt-private-key/--no-encrypt-private-key",
        help="Encrypt the private key with a passphrase",
    ),
    passphrase: str | None = typer.Option(None, help="Private key passphrase", hide_input=True),
    passphrase_env: str | None = typer.Option(None, help="Environment variable containing the passphrase"),
    passphrase_file: Path | None = typer.Option(None, help="File containing the passphrase"),
) -> None:
    output_dir = safe_output_path(output_dir)
    resolved_passphrase = resolve_passphrase(
        passphrase=passphrase,
        passphrase_env=passphrase_env,
        passphrase_file=passphrase_file,
    )
    if encrypt_private_key and resolved_passphrase is None:
        raise typer.BadParameter("Encrypted key generation requires a passphrase source")
    if not encrypt_private_key and resolved_passphrase is not None:
        raise typer.BadParameter("Passphrase was provided but --encrypt-private-key is disabled")
    payload = generate_keypair(
        output_dir=output_dir,
        private_name=private_name,
        public_name=public_name,
        passphrase=resolved_passphrase,
    )
    typer.echo(json.dumps(payload, indent=2))


@authority_app.command("sign")
def authority_sign(
    input_json: Path,
    privkey: Path = typer.Option(..., help="Path to Ed25519 private key PEM"),
    output: Path = typer.Option(Path("results/mdmp_report.signed.mdmp"), help="Output signed artifact"),
    signed_by: str = typer.Option("MDMP-Authority-v1", help="Signer identity label"),
    key_id: str = typer.Option("mdmp_pub_v1", help="Public key identifier"),
    expires_days: int | None = typer.Option(None, help="Optional expiry in days if card has no expires field"),
    passphrase: str | None = typer.Option(None, help="Private key passphrase", hide_input=True),
    passphrase_env: str | None = typer.Option(None, help="Environment variable containing the passphrase"),
    passphrase_file: Path | None = typer.Option(None, help="File containing the passphrase"),
) -> None:
    payload = json.loads(input_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise typer.BadParameter("Input card must be a JSON object")
    signer = MDMPSigner(
        privkey,
        signed_by=signed_by,
        key_id=key_id,
        private_key_passphrase=resolve_passphrase(
            passphrase=passphrase,
            passphrase_env=passphrase_env,
            passphrase_file=passphrase_file,
        ),
    )
    signed = signer.sign_card(payload, expires_days=expires_days)
    output = prepare_output_path(output)
    output.write_text(json.dumps(signed, indent=2), encoding="utf-8")
    typer.echo(f"Signed artifact: {output}")
    typer.echo(f"Signed by: {signed.get('signed_by')} ({signed.get('key_id')})")


@authority_app.command("delegate")
def authority_delegate(
    delegate_id: str = typer.Option(..., help="Delegate identifier"),
    delegate_name: str = typer.Option(..., help="Delegate display name"),
    delegate_pubkey: Path = typer.Option(..., help="Path to delegate Ed25519 public key PEM"),
    grades: list[str] = typer.Option(
        ["draft", "research_grade"],
        "--grades",
        help="Allowed grades for delegate (repeat option for multiple values)",
    ),
    valid_days: int = typer.Option(365, help="Delegate certificate validity period"),
    max_expires_days: int = typer.Option(365, help="Max card expiry window delegate can issue"),
    require_consent: bool = typer.Option(
        False,
        "--require-consent/--no-require-consent",
        help="Require consent field in delegate-signed cards",
    ),
    flavors: list[str] = typer.Option(
        [],
        "--flavors",
        help="Allowed flavors for delegate-signed cards (repeat for multiple)",
    ),
    privkey: Path = typer.Option(..., help="Root authority private key PEM"),
    output: Path = typer.Option(Path("certs/delegate.cert"), help="Output delegate certificate"),
    signed_by: str = typer.Option("MDMP-Authority-v1", help="Root signer identity"),
    key_id: str = typer.Option("mdmp_pub_v1", help="Root public key identifier"),
    passphrase: str | None = typer.Option(None, help="Root private key passphrase", hide_input=True),
    passphrase_env: str | None = typer.Option(None, help="Environment variable containing the passphrase"),
    passphrase_file: Path | None = typer.Option(None, help="File containing the passphrase"),
) -> None:
    signer = MDMPSigner(
        privkey,
        signed_by=signed_by,
        key_id=key_id,
        private_key_passphrase=resolve_passphrase(
            passphrase=passphrase,
            passphrase_env=passphrase_env,
            passphrase_file=passphrase_file,
        ),
    )
    issuer = DelegateIssuer(signer)
    constraints = DelegateConstraints(
        max_expires_days=max_expires_days,
        require_consent_field=require_consent,
        allowed_flavors=flavors,
        allowed_grades=grades,
    )
    cert = issuer.issue(
        delegate_id=delegate_id,
        delegate_name=delegate_name,
        delegate_pubkey_path=delegate_pubkey,
        allowed_grades=grades,
        valid_days=valid_days,
        constraints=constraints,
    )
    output = prepare_output_path(output)
    output.write_text(json.dumps(cert, indent=2), encoding="utf-8")
    typer.echo(f"Delegate certificate: {output}")
    typer.echo(f"Delegate: {delegate_id} ({delegate_name})")


@trust_app.command("init")
def trust_init_command(
    trust_store: Path = typer.Option(
        Path("trust/mdmp_trust_store.json"),
        help="Trust store JSON path",
    ),
    key_id: str = typer.Option("mdmp_pub_v1", help="Trusted key id"),
    public_key: Path = typer.Option(..., help="Public key PEM path"),
    set_active: bool = typer.Option(True, "--set-active/--no-set-active", help="Set this key as active"),
    signed_by: str | None = typer.Option(None, help="Expected signed_by label for this key"),
) -> None:
    payload = trust_store_init(
        trust_store,
        key_id=key_id,
        public_key_path=public_key,
        set_active=set_active,
        signed_by=signed_by,
    )
    typer.echo(f"Initialized trust store: {trust_store}")
    typer.echo(f"Trusted keys: {len(payload.get('trusted_keys', {}))}")
    typer.echo(f"Active key: {payload.get('active_key_id')}")


@trust_app.command("add-key")
def trust_add_key_command(
    trust_store: Path = typer.Option(
        Path("trust/mdmp_trust_store.json"),
        help="Trust store JSON path",
    ),
    key_id: str = typer.Option(..., help="Trusted key id"),
    public_key: Path = typer.Option(..., help="Public key PEM path"),
    set_active: bool = typer.Option(False, "--set-active/--no-set-active", help="Set this key as active"),
    signed_by: str | None = typer.Option(None, help="Expected signed_by label for this key"),
) -> None:
    payload = trust_add_key(
        trust_store,
        key_id=key_id,
        public_key_path=public_key,
        set_active=set_active,
        signed_by=signed_by,
    )
    typer.echo(f"Updated trust store: {trust_store}")
    typer.echo(f"Trusted keys: {len(payload.get('trusted_keys', {}))}")
    typer.echo(f"Active key: {payload.get('active_key_id')}")


@trust_app.command("revoke-key")
def trust_revoke_key_command(
    key_id: str = typer.Argument(..., help="Key id to revoke"),
    trust_store: Path = typer.Option(
        Path("trust/mdmp_trust_store.json"),
        help="Trust store JSON path",
    ),
    reason: str = typer.Option("revoked", help="Revocation reason"),
) -> None:
    payload = trust_revoke_key(
        trust_store,
        key_id=key_id,
        reason=reason,
    )
    typer.echo(f"Revoked key: {key_id}")
    typer.echo(f"Revoked keys: {len(payload.get('revoked_keys', {}))}")


@trust_app.command("unrevoke-key")
def trust_unrevoke_key_command(
    key_id: str = typer.Argument(..., help="Key id to un-revoke"),
    trust_store: Path = typer.Option(
        Path("trust/mdmp_trust_store.json"),
        help="Trust store JSON path",
    ),
) -> None:
    payload = trust_unrevoke_key(trust_store, key_id=key_id)
    typer.echo(f"Unrevoked key: {key_id}")
    typer.echo(f"Revoked keys: {len(payload.get('revoked_keys', {}))}")


@trust_app.command("revoke-delegate")
def trust_revoke_delegate_command(
    delegate_id: str = typer.Argument(..., help="Delegate id to revoke"),
    trust_store: Path = typer.Option(
        Path("trust/mdmp_trust_store.json"),
        help="Trust store JSON path",
    ),
    reason: str = typer.Option("revoked", help="Revocation reason"),
) -> None:
    payload = trust_revoke_delegate(
        trust_store,
        delegate_id=delegate_id,
        reason=reason,
    )
    typer.echo(f"Revoked delegate: {delegate_id}")
    typer.echo(f"Revoked delegates: {len(payload.get('revoked_delegates', {}))}")


@trust_app.command("unrevoke-delegate")
def trust_unrevoke_delegate_command(
    delegate_id: str = typer.Argument(..., help="Delegate id to un-revoke"),
    trust_store: Path = typer.Option(
        Path("trust/mdmp_trust_store.json"),
        help="Trust store JSON path",
    ),
) -> None:
    payload = trust_unrevoke_delegate(trust_store, delegate_id=delegate_id)
    typer.echo(f"Unrevoked delegate: {delegate_id}")
    typer.echo(f"Revoked delegates: {len(payload.get('revoked_delegates', {}))}")


@trust_app.command("show")
def trust_show(
    trust_store: Path = typer.Option(
        Path("trust/mdmp_trust_store.json"),
        help="Trust store JSON path",
    ),
) -> None:
    payload = load_trust_store(trust_store)
    typer.echo(json.dumps(payload, indent=2))


@app.command("validate")
def validate(
    contract_path: Path,
    dataset_csv: Path,
    output_json: Path = typer.Option(Path("results/mdmp_report.json"), help="Output report JSON"),
    fingerprint_store: Path | None = typer.Option(
        None,
        help="Optional fingerprint store JSON for stale/expiry checks",
    ),
    policy: Path | None = typer.Option(
        None,
        help="Optional policy YAML/JSON path to evaluate and enforce",
    ),
    fail_on_policy: bool = typer.Option(
        False,
        "--fail-on-policy/--no-fail-on-policy",
        help="Exit non-zero if policy evaluation fails",
    ),
) -> None:
    contract = load_contract(contract_path)
    df = pd.read_csv(dataset_csv)
    result = ContractRunner(contract).run(df)
    payload = result.to_dict()
    payload.setdefault("spec_version", "1.0")
    payload.setdefault("mdmp_object", "validation_report")
    payload["effective_grade"] = payload.get("grade")
    payload["effective_grade_reason"] = "base_grade"

    if fingerprint_store is not None:
        store = FingerprintStore(fingerprint_store)
        stored = store.get_by_dataset(str(dataset_csv))
        if stored is not None:
            staleness = check_fingerprint(stored, data_path=dataset_csv)
            payload["staleness"] = staleness
            warnings = payload.setdefault("warnings", [])
            if staleness.get("status") == "stale":
                reason = staleness.get("stale_reason", "unknown")
                warnings.append(
                    f"Dataset fingerprint is stale ({reason}). Re-grade required before AI training."
                )
                payload["effective_grade"] = "draft"
                payload["effective_grade_reason"] = f"stale_fingerprint_{reason}"

    policy_spec = load_policy(policy) if policy is not None else default_policy()
    policy_eval = evaluate_policy(payload, policy_spec)
    payload["policy_evaluation"] = policy_eval
    effective_grade, effective_reason = apply_policy_effects(payload, policy_eval, policy_spec)
    payload["effective_grade"] = effective_grade
    payload["effective_grade_reason"] = effective_reason

    output_json = prepare_output_path(output_json)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    typer.echo(f"Saved report: {output_json}")
    typer.echo(f"Grade: {payload.get('effective_grade')} (base={result.grade}) | Compliance: {result.compliance_score}%")
    if payload.get("staleness", {}).get("status") == "stale":
        typer.echo(f"Staleness: STALE ({payload['staleness'].get('stale_reason')})")
    typer.echo(f"Policy: {'PASS' if policy_eval.get('passed') else 'FAIL'}")
    if policy_eval.get("failed_checks"):
        typer.echo(f"Policy failed checks: {', '.join(policy_eval.get('failed_checks', []))}")
    if fail_on_policy and not policy_eval.get("passed"):
        raise typer.Exit(code=2)


@app.command("grade")
def grade(contract_path: Path, dataset_csv: Path) -> None:
    contract = load_contract(contract_path)
    df = pd.read_csv(dataset_csv)
    result = ContractRunner(contract).run(df)
    typer.echo(result.grade)


@app.command("policy-template")
def policy_template(
    output: Path = typer.Option(Path("mdmp_policy.yaml"), help="Policy file output path"),
) -> None:
    output = safe_output_path(output)
    save_policy(output, default_policy())
    typer.echo(f"Saved policy template: {output}")


@app.command("policy-eval")
def policy_eval(
    policy_path: Path,
    input_json: Path,
    output_json: Path | None = typer.Option(None, help="Optional output JSON path"),
) -> None:
    policy_spec = load_policy(policy_path)
    payload = json.loads(input_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise typer.BadParameter("input_json must contain a JSON object")
    result = evaluate_policy(payload, policy_spec)
    typer.echo(f"Policy result: {'PASS' if result.get('passed') else 'FAIL'}")
    if result.get("failed_checks"):
        typer.echo(f"Failed checks: {', '.join(result.get('failed_checks', []))}")
    if output_json is not None:
        output_json = prepare_output_path(output_json)
        output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        typer.echo(f"Saved policy evaluation: {output_json}")


@app.command("spec-version")
def spec_version(
    artifact_json: Path,
    output_json: Path | None = typer.Option(None, help="Optional output JSON path"),
) -> None:
    payload = load_json(artifact_json)
    info = detect_version(payload)
    typer.echo(f"spec_version: {info['current_version']}")
    typer.echo(f"target_version: {info['target_version']}")
    typer.echo(f"up_to_date: {info['up_to_date']}")
    if info["migration_available"]:
        typer.echo(f"migration_path: {', '.join(info['migration_path'])}")
    if output_json is not None:
        output_json = prepare_output_path(output_json)
        output_json.write_text(json.dumps(info, indent=2), encoding="utf-8")
        typer.echo(f"Saved version info: {output_json}")


@app.command("migrate")
def migrate_artifact(
    artifact_json: Path,
    to: str = typer.Option(CURRENT_SPEC_VERSION, "--to", help="Target spec version"),
    output: Path | None = typer.Option(None, help="Output artifact path"),
    in_place: bool = typer.Option(False, "--in-place", help="Rewrite source artifact"),
    backup: bool = typer.Option(False, "--backup", help="Create .bak backup when in-place"),
) -> None:
    if output is not None:
        output = safe_output_path(output)
    result = migrate_file(
        artifact_json,
        target_version=to,
        destination=output,
        in_place=in_place,
        backup=backup,
    )
    typer.echo(f"Source: {result.source}")
    typer.echo(f"Destination: {result.destination}")
    typer.echo(f"Version: {result.before_version} -> {result.after_version}")
    typer.echo(f"Changed: {result.changed}")


@app.command("migrate-dir")
def migrate_directory(
    directory: Path,
    to: str = typer.Option(CURRENT_SPEC_VERSION, "--to", help="Target spec version"),
    backup: bool = typer.Option(True, "--backup/--no-backup", help="Create .bak backups"),
    strict: bool = typer.Option(False, "--strict/--no-strict", help="Fail on first invalid JSON artifact"),
) -> None:
    total = 0
    migrated_count = 0
    skipped = 0
    for artifact_path in iter_json_files(directory):
        total += 1
        try:
            result = migrate_file(
                artifact_path,
                target_version=to,
                in_place=True,
                backup=backup,
            )
            if result.changed:
                migrated_count += 1
            else:
                skipped += 1
        except Exception as exc:
            skipped += 1
            typer.echo(f"Skip {artifact_path}: {exc}")
            if strict:
                raise typer.Exit(code=1)
    typer.echo(f"Processed: {total}")
    typer.echo(f"Migrated: {migrated_count}")
    typer.echo(f"Skipped: {skipped}")


@app.command("fingerprint")
def fingerprint(dataset_csv: Path) -> None:
    df = pd.read_csv(dataset_csv)
    typer.echo(f"sha256:{dataframe_fingerprint(df)}")


@app.command("fingerprint-record")
def fingerprint_record(
    dataset_path: Path,
    output_json: Path = typer.Option(Path("results/fingerprint.json"), help="Output fingerprint JSON"),
    expires_days: int = typer.Option(365, help="Validity period in days"),
    fingerprint_store: Path | None = typer.Option(
        None,
        help="Optional fingerprint store JSON to upsert this record",
    ),
) -> None:
    record = compute_fingerprint(dataset_path, expires_days=expires_days)
    output_json = prepare_output_path(output_json)
    output_json.write_text(json.dumps(record, indent=2), encoding="utf-8")
    typer.echo(f"Saved fingerprint record: {output_json}")
    typer.echo(f"Fingerprint: {record['fingerprint']}")
    typer.echo(f"Expires: {record['expires']}")

    if fingerprint_store is not None:
        store = FingerprintStore(fingerprint_store)
        store.upsert(dataset_path=str(dataset_path), record=record)
        typer.echo(f"Updated fingerprint store: {fingerprint_store}")


@app.command("fingerprint-check")
def fingerprint_check(
    fingerprint_json: Path,
    dataset_path: Path,
    output_json: Path | None = typer.Option(None, help="Optional output path for check result"),
) -> None:
    stored = json.loads(fingerprint_json.read_text(encoding="utf-8"))
    checked = check_fingerprint(stored, data_path=dataset_path)
    typer.echo(f"Fingerprint: {checked.get('fingerprint')}")
    typer.echo(f"Status: {str(checked.get('status', '')).upper()}")
    typer.echo(f"Reason: {checked.get('stale_reason')}")
    typer.echo(f"Created: {checked.get('created')}")
    typer.echo(f"Expires: {checked.get('expires')}")
    typer.echo(f"Checked at: {checked.get('checked_at')}")

    if output_json is not None:
        output_json = prepare_output_path(output_json)
        output_json.write_text(json.dumps(checked, indent=2), encoding="utf-8")
        typer.echo(f"Saved check result: {output_json}")


@app.command("delegate-sign")
def delegate_sign(
    input_json: Path,
    privkey: Path = typer.Option(..., help="Delegate Ed25519 private key PEM"),
    cert: Path = typer.Option(..., help="Delegate certificate JSON"),
    root_public_key: Path | None = typer.Option(
        None,
        help="Optional root public key PEM for delegate certificate verification",
    ),
    trust_store: Path | None = typer.Option(
        None,
        help="Optional trust store JSON for delegate certificate verification",
    ),
    output: Path = typer.Option(Path("results/mdmp_report.signed.mdmp"), help="Output signed artifact"),
    expires_days: int | None = typer.Option(
        None,
        help="Optional default expires window for delegate cards if field is missing",
    ),
    passphrase: str | None = typer.Option(None, help="Delegate private key passphrase", hide_input=True),
    passphrase_env: str | None = typer.Option(None, help="Environment variable containing the passphrase"),
    passphrase_file: Path | None = typer.Option(None, help="File containing the passphrase"),
) -> None:
    payload = json.loads(input_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise typer.BadParameter("Input card must be a JSON object")
    cert_payload = json.loads(cert.read_text(encoding="utf-8"))
    if not isinstance(cert_payload, dict):
        raise typer.BadParameter("Certificate must be a JSON object")

    signer = DelegateSigner(
        privkey,
        cert_payload,
        root_public_key_path=root_public_key,
        trust_store_path=trust_store,
        delegate_private_key_passphrase=resolve_passphrase(
            passphrase=passphrase,
            passphrase_env=passphrase_env,
            passphrase_file=passphrase_file,
        ),
    )
    signed = signer.sign_card(payload, default_expires_days=expires_days)
    output = prepare_output_path(output)
    output.write_text(json.dumps(signed, indent=2), encoding="utf-8")
    typer.echo(f"Delegate-signed artifact: {output}")
    typer.echo(f"Delegate: {signed.get('issued_by_delegate')}")


@app.command("verify")
def verify(
    signed_card: Path,
    public_key: Path | None = typer.Option(None, help="Optional public key PEM (defaults to bundled key)"),
    trust_store: Path | None = typer.Option(
        None,
        help="Optional trust store JSON (enables key rotation/revocation checks)",
    ),
    dataset: Path | None = typer.Option(None, help="Optional dataset file to verify fingerprint match"),
    cert: Path | None = typer.Option(
        None,
        help="Optional delegate certificate JSON for delegated trust-chain verification",
    ),
    output_json: Path | None = typer.Option(None, help="Optional output verification JSON"),
) -> None:
    payload = json.loads(signed_card.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise typer.BadParameter("Signed card must be a JSON object")

    if cert is not None:
        cert_payload = json.loads(cert.read_text(encoding="utf-8"))
        if not isinstance(cert_payload, dict):
            raise typer.BadParameter("Certificate must be a JSON object")
        verifier = DelegateVerifier(public_key, trust_store_path=trust_store)
        result = verifier.verify(payload, cert_payload, dataset_path=dataset)
        status = "TRUSTED" if result.get("valid") else "UNTRUSTED"
        typer.echo("MDMP Trust Chain Verification")
        typer.echo("=" * 40)
        for idx, step in enumerate(result.get("chain", []), start=1):
            mark = "✓" if step.get("valid") else "✗"
            typer.echo(f"{mark} [{idx}] {step.get('step')}: {step.get('detail')}")
        typer.echo("=" * 40)
        typer.echo(f"Result: {status}")
        typer.echo(f"Delegate: {result.get('delegate_id')}")
        typer.echo(f"Grade: {result.get('grade')}")
        if result.get("error"):
            typer.echo(f"Error: {result.get('error')}")
    else:
        verifier = MDMPVerifier(public_key, trust_store_path=trust_store)
        result = verifier.verify(payload, dataset_path=dataset)

        status = "VALID" if result.get("valid") else "INVALID"
        tamper = "CLEAN" if not result.get("tampered") else "TAMPER DETECTED"
        typer.echo(f"Signature: {status}")
        typer.echo(f"Issued by: {result.get('issued_by')} ({result.get('key_id')})")
        typer.echo(f"Signed at: {result.get('signed_at')}")
        typer.echo(f"Grade: {result.get('grade')}")
        typer.echo(f"Expires: {result.get('expires')} | Expired: {result.get('expired')}")
        typer.echo(f"Fingerprint match: {result.get('fingerprint_matches')}")
        typer.echo(f"Tamper check: {tamper}")
        if result.get("error"):
            typer.echo(f"Error: {result.get('error')}")

    if output_json is not None:
        output_json = prepare_output_path(output_json)
        output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        typer.echo(f"Saved verification output: {output_json}")


@app.command("report")
def report(
    report_json: Path,
    output_html: Path = typer.Option(Path("results/mdmp_dashboard.html"), help="Output dashboard HTML"),
    title: str = typer.Option("MDMP Dashboard", help="HTML title"),
) -> None:
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    html = build_dashboard_html(payload, title=title)
    output_html = prepare_output_path(output_html)
    output_html.write_text(html, encoding="utf-8")
    typer.echo(f"Saved dashboard: {output_html}")


@app.command("schema-export")
def schema_export(
    contract_path: Path,
    format: str = typer.Option("json-schema", help="Export format: json-schema|frictionless"),
    output: Path = typer.Option(Path("results/mdmp_schema.json"), help="Output schema JSON"),
) -> None:
    contract = load_contract(contract_path)
    format_key = format.strip().lower()
    if format_key == "json-schema":
        payload = contract_to_json_schema(contract)
    elif format_key == "frictionless":
        payload = contract_to_frictionless_schema(contract)
    else:
        raise typer.BadParameter("format must be one of: json-schema, frictionless")
    output = prepare_output_path(output)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    typer.echo(f"Saved schema export: {output}")
    typer.echo(f"Format: {format_key}")


@app.command("prov-export")
def prov_export(
    card_json: Path,
    output: Path = typer.Option(Path("results/mdmp_prov.jsonld"), help="Output PROV JSON-LD file"),
) -> None:
    payload = json.loads(card_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise typer.BadParameter("card_json must contain a JSON object")
    prov_payload = card_to_prov(payload)
    output = prepare_output_path(output)
    output.write_text(json.dumps(prov_payload, indent=2), encoding="utf-8")
    typer.echo(f"Saved PROV export: {output}")


@app.command("diff")
def diff(
    baseline_csv: Path,
    candidate_csv: Path,
    contract_path: Path | None = typer.Option(None, help="Optional contract for bounds-aware diff"),
    output_json: Path = typer.Option(Path("results/mdmp_diff.json"), help="Output diff JSON"),
) -> None:
    baseline_df = pd.read_csv(baseline_csv)
    candidate_df = pd.read_csv(candidate_csv)
    contract = load_contract(contract_path) if contract_path is not None else None
    payload = compare_datasets(baseline_df, candidate_df, contract=contract)
    payload["baseline_csv"] = str(baseline_csv)
    payload["candidate_csv"] = str(candidate_csv)

    output_json = prepare_output_path(output_json)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    typer.echo(f"Saved diff report: {output_json}")
    typer.echo(f"Changed: {payload['has_changes']}")
    typer.echo(f"Row delta: {payload['row_count']['delta']}")


@app.command("drift")
def drift(
    baseline_csv: Path,
    candidate_csv: Path,
    columns: list[str] = typer.Option([], "--columns", help="Optional numeric columns to evaluate"),
    contract: Path | None = typer.Option(None, help="Optional contract for schema/bounds diff context"),
    output_json: Path | None = typer.Option(None, help="Optional output JSON path"),
    fail_on: str | None = typer.Option(
        None,
        help="Fail when overall severity reaches threshold: ok|warn|critical",
    ),
) -> None:
    threshold = None
    if fail_on is not None:
        candidate_threshold = fail_on.strip().lower()
        if candidate_threshold not in valid_severities():
            allowed = ", ".join(valid_severities())
            raise typer.BadParameter(f"--fail-on must be one of: {allowed}")
        threshold = candidate_threshold

    report = compute_drift(
        str(baseline_csv),
        str(candidate_csv),
        columns=columns or None,
    )
    typer.echo(format_drift_report(report))

    payload = report.to_dict()
    if contract is not None:
        baseline_df = pd.read_csv(baseline_csv)
        candidate_df = pd.read_csv(candidate_csv)
        payload["contract_context"] = compare_datasets(
            baseline_df,
            candidate_df,
            contract=load_contract(contract),
        )

    if output_json is not None:
        output_json = prepare_output_path(output_json)
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        typer.echo(f"Saved drift report: {output_json}")

    if threshold is not None and severity_at_or_above(report.overall_severity, threshold):
        raise typer.Exit(code=1)


@app.command("compare")
def compare(
    baseline_report_json: Path,
    candidate_report_json: Path,
    output_json: Path = typer.Option(Path("results/mdmp_compare.json"), help="Output comparison JSON"),
) -> None:
    baseline = json.loads(baseline_report_json.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_report_json.read_text(encoding="utf-8"))
    payload = compare_reports(baseline=baseline, candidate=candidate)
    output_json = prepare_output_path(output_json)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    typer.echo(f"Saved comparison: {output_json}")
    typer.echo(
        f"Grade delta rank: {payload['delta']['grade_rank']} | Score delta: {payload['delta']['compliance_score']}"
    )


@app.command("audit")
def audit(
    report_json: Path,
    lineage_card: Path | None = typer.Option(None, help="Optional lineage card (.yaml/.json)"),
    fingerprint_json: Path | None = typer.Option(None, help="Optional fingerprint record JSON"),
    registry_json: Path | None = typer.Option(None, help="Optional registry JSON"),
    validated_by: str = typer.Option("unknown", help="Actor/user that performed validation"),
    output_json: Path = typer.Option(Path("results/mdmp_audit.json"), help="Output audit JSON"),
    output_html: Path = typer.Option(Path("results/mdmp_audit.html"), help="Output audit HTML"),
) -> None:
    payload = build_audit_from_sources(
        report_json=report_json,
        lineage_card=lineage_card,
        fingerprint_json=fingerprint_json,
        registry_json=registry_json,
        validated_by=validated_by,
    )

    output_json = prepare_output_path(output_json)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_html = prepare_output_path(output_html)
    output_html.write_text(build_audit_html(payload), encoding="utf-8")

    typer.echo(f"Saved audit JSON: {output_json}")
    typer.echo(f"Saved audit HTML: {output_html}")
    typer.echo(f"Audit fingerprint: {payload.get('audit_fingerprint')}")


@app.command("audit-bundle")
def audit_bundle(
    report_json: Path,
    privkey: Path = typer.Option(..., help="Authority private key PEM for bundle signing"),
    lineage_json: Path | None = typer.Option(None, help="Optional lineage card JSON/YAML"),
    fingerprint_json: Path | None = typer.Option(None, help="Optional fingerprint JSON"),
    registry_json: Path | None = typer.Option(None, help="Optional registry JSON"),
    validated_by: str = typer.Option("unknown", help="Actor/user that performed validation"),
    signed_by: str = typer.Option("MDMP-Authority-v1", help="Signer identity label"),
    key_id: str = typer.Option("mdmp_pub_v1", help="Public key identifier"),
    expires_days: int | None = typer.Option(None, help="Optional bundle expiry in days"),
    output_json: Path = typer.Option(Path("results/mdmp_audit_bundle.signed.json"), help="Output signed bundle JSON"),
) -> None:
    bundle_payload = build_bundle_from_files(
        report_json=report_json,
        validated_by=validated_by,
        lineage_json=lineage_json,
        fingerprint_json=fingerprint_json,
        registry_json=registry_json,
    )
    signer = MDMPSigner(privkey, signed_by=signed_by, key_id=key_id)
    signed_bundle = signer.sign_card(bundle_payload, expires_days=expires_days)
    output_json = prepare_output_path(output_json)
    output_json.write_text(json.dumps(signed_bundle, indent=2), encoding="utf-8")
    typer.echo(f"Saved signed audit bundle: {output_json}")
    typer.echo(f"Bundle hash: {bundle_payload.get('bundle_hash')}")


@app.command("audit-bundle-verify")
def audit_bundle_verify(
    bundle_json: Path,
    public_key: Path | None = typer.Option(None, help="Optional public key PEM (defaults to bundled key)"),
    trust_store: Path | None = typer.Option(None, help="Optional trust store JSON"),
    output_json: Path | None = typer.Option(None, help="Optional output JSON path"),
) -> None:
    payload = json.loads(bundle_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise typer.BadParameter("bundle_json must contain a JSON object")

    verify_result = MDMPVerifier(public_key, trust_store_path=trust_store).verify(payload)
    unsigned = normalize_unsigned_payload(payload)
    integrity_result = verify_bundle_integrity(unsigned)
    passed = bool(verify_result.get("valid") and integrity_result.get("passed"))
    result = {
        "passed": passed,
        "signature": verify_result,
        "integrity": integrity_result,
    }
    typer.echo(f"Bundle verification: {'PASS' if passed else 'FAIL'}")
    if output_json is not None:
        output_json = prepare_output_path(output_json)
        output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        typer.echo(f"Saved bundle verification: {output_json}")


@app.command("bias-run")
def bias_run(
    report_json: Path,
    output_json: Path = typer.Option(Path("results/mdmp_bias_hooks.json"), help="Output hook results JSON"),
) -> None:
    report_payload = json.loads(report_json.read_text(encoding="utf-8"))
    payload = run_bias_hooks(report_payload)
    output_json = prepare_output_path(output_json)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    typer.echo(f"Saved bias hook output: {output_json}")
    typer.echo(f"Hooks executed: {payload.get('hook_count', 0)}")


@app.command("certify")
def certify(
    report_json: Path,
    issued_by: str = typer.Option(..., help="Issuer id (team/user/system)"),
    level: str = typer.Option("research_grade", help="Certificate level"),
    output_json: Path = typer.Option(Path("results/mdmp_certificate.json"), help="Output certificate JSON"),
) -> None:
    report_payload = json.loads(report_json.read_text(encoding="utf-8"))
    payload = create_certificate(report_payload, issued_by=issued_by, level=level)
    output_json = prepare_output_path(output_json)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    typer.echo(f"Saved certificate: {output_json}")
    typer.echo(f"Signature: {payload.get('signature_sha256')}")


@app.command("synthetic-track")
def synthetic_track(
    source_fingerprint: str = typer.Option(..., help="Source dataset fingerprint sha256:..."),
    generator: str = typer.Option(..., help="Generator model/system"),
    method: str = typer.Option("ctgan", help="Generation method"),
    privacy_epsilon: float | None = typer.Option(None, help="Optional DP epsilon"),
    notes: str | None = typer.Option(None, help="Optional notes"),
    output_json: Path = typer.Option(Path("results/mdmp_synthetic_metadata.json"), help="Output metadata JSON"),
) -> None:
    payload = build_synthetic_metadata(
        generator=generator,
        source_fingerprint=source_fingerprint,
        method=method,
        privacy_epsilon=privacy_epsilon,
        notes=notes,
    )
    output_json = prepare_output_path(output_json)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    typer.echo(f"Saved synthetic metadata: {output_json}")


@app.command("llm-card")
def llm_card(
    model_name: str = typer.Option(..., help="Model name"),
    corpora_json: Path = typer.Option(..., help="JSON file with list of corpora entries"),
    tokenizer: str = typer.Option(..., help="Tokenizer id"),
    pretraining_tokens: int = typer.Option(..., help="Pretraining token count"),
    fine_tune_tokens: int = typer.Option(0, help="Fine-tune token count"),
    output: Path = typer.Option(Path("results/mdmp_llm_card.yaml"), help="Output LLM provenance card"),
) -> None:
    corpora = json.loads(corpora_json.read_text(encoding="utf-8"))
    if not isinstance(corpora, list):
        raise typer.BadParameter("corpora_json must contain a JSON list")
    payload = build_llm_training_card(
        model_name=model_name,
        corpora=corpora,
        tokenizer=tokenizer,
        pretraining_tokens=pretraining_tokens,
        fine_tune_tokens=fine_tune_tokens,
    )
    output = prepare_output_path(output)
    if output.suffix.lower() in {".yaml", ".yml"}:
        output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    else:
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    typer.echo(f"Saved LLM provenance card: {output}")


@app.command("lineage-card")
def lineage_card(
    model: str = typer.Option(..., help="Model name"),
    dataset: Path = typer.Option(..., help="Dataset CSV path"),
    contract: Path = typer.Option(..., help="Contract YAML path"),
    output: Path = typer.Option(Path("results/mdmp_model_card.yaml"), help="Output model card (.yaml/.json)"),
    expires_days: int = typer.Option(365, help="Validity period for dataset fingerprint in days"),
    fingerprint_store: Path | None = typer.Option(
        None,
        help="Optional fingerprint store JSON to persist lineage dataset fingerprints",
    ),
) -> None:
    output = safe_output_path(output)
    tracker = LineageTracker()
    record = tracker.register_dataset(dataset, contract=contract, expires_days=expires_days)
    tracker.attach_to_model(model)
    card = tracker.export_card(output)

    typer.echo(f"Dataset registered: {record['fingerprint']} ({record['grade']})")
    typer.echo(f"Saved model card: {output}")
    typer.echo(yaml.safe_dump(card, sort_keys=False))

    if fingerprint_store is not None:
        store = FingerprintStore(fingerprint_store)
        store.upsert(
            dataset_path=str(dataset),
            record={
                "fingerprint": record["fingerprint"],
                "created": record["date_validated"],
                "expires": record["expires"],
                "source_path": str(dataset),
                "status": record.get("status", "valid"),
                "stale_reason": record.get("stale_reason"),
            },
        )
        typer.echo(f"Updated fingerprint store: {fingerprint_store}")


@app.command("lineage-card-refresh")
def lineage_card_refresh(
    card_path: Path,
    output: Path | None = typer.Option(None, help="Optional output path (default: overwrite input card)"),
    data_dir: Path | None = typer.Option(None, help="Optional data directory for resolving relative dataset paths"),
) -> None:
    card = load_lineage_card(card_path)
    card.refresh(data_dir=data_dir)

    destination = safe_output_path(output) if output is not None else card_path
    payload = card.export(destination)
    stale = payload.get("model", {}).get("stale_datasets", [])
    typer.echo(f"Model: {payload.get('model', {}).get('name')}")
    typer.echo(f"Lineage status: {payload.get('model', {}).get('lineage_status')}")
    typer.echo(f"Stale datasets: {len(stale)}")
    typer.echo(f"Saved lineage card: {destination}")


@registry_app.command("init")
def registry_init(
    registry: Path = typer.Option(Path("registry/mdmp_registry.json"), help="Registry JSON path"),
) -> None:
    registry = safe_output_path(registry)
    payload = init_registry(registry)
    typer.echo(f"Initialized registry: {registry}")
    typer.echo(f"Version: {payload['version']}")


@registry_app.command("push")
def registry_push(
    registry: Path = typer.Option(Path("registry/mdmp_registry.json"), help="Registry JSON path"),
    fingerprint: str | None = typer.Option(None, help="Dataset fingerprint (sha256:...)"),
    report: Path | None = typer.Option(None, help="Optional MDMP report JSON to ingest"),
    grade: str | None = typer.Option(None, help="Optional grade override"),
    source: str | None = typer.Option(None, help="Optional source label (dataset/report name)"),
    visibility: str = typer.Option("private", help="Record visibility: private|public"),
    model_id: list[str] = typer.Option([], help="Model id(s) using this dataset"),
    expires: str | None = typer.Option(None, help="Optional expires timestamp"),
    status: str | None = typer.Option(None, help="Optional status override: valid|stale"),
    stale_reason: str | None = typer.Option(None, help="Optional stale reason"),
    metadata_json: Path | None = typer.Option(None, help="Optional metadata JSON file"),
) -> None:
    registry = safe_output_path(registry)
    report_payload = None
    if report is not None:
        report_payload = json.loads(report.read_text(encoding="utf-8"))
    metadata = None
    if metadata_json is not None:
        metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise typer.BadParameter("metadata_json must contain a JSON object")

    record = upsert_record(
        registry,
        fingerprint=fingerprint,
        report=report_payload,
        grade=grade,
        source=source,
        visibility=visibility,
        used_in_models=model_id,
        expires=expires,
        status=status,
        stale_reason=stale_reason,
        metadata=metadata,
    )
    typer.echo(f"Registry updated: {registry}")
    typer.echo(f"Fingerprint: {record['fingerprint']}")
    typer.echo(f"Grade: {record['grade']} | Status: {record['status']} | Visibility: {record['visibility']}")


@registry_app.command("lookup")
def registry_lookup(
    fingerprint: str,
    registry: Path = typer.Option(Path("registry/mdmp_registry.json"), help="Registry JSON path"),
    output_json: Path | None = typer.Option(None, help="Optional output JSON path"),
) -> None:
    registry = safe_output_path(registry)
    record = lookup_record(registry, fingerprint)
    if record is None:
        typer.echo(f"Fingerprint not found: {fingerprint}")
        raise typer.Exit(code=1)
    text = json.dumps(record, indent=2)
    typer.echo(text)
    if output_json is not None:
        output_json = prepare_output_path(output_json)
        output_json.write_text(text, encoding="utf-8")
        typer.echo(f"Saved lookup: {output_json}")


@registry_app.command("list")
def registry_list(
    registry: Path = typer.Option(Path("registry/mdmp_registry.json"), help="Registry JSON path"),
    grade: str | None = typer.Option(None, help="Filter by grade"),
    visibility: str | None = typer.Option(None, help="Filter by visibility"),
    status: str | None = typer.Option(None, help="Filter by status"),
    limit: int = typer.Option(20, help="Maximum rows"),
    output_json: Path | None = typer.Option(None, help="Optional output JSON path"),
) -> None:
    registry = safe_output_path(registry)
    records = list_records(
        registry,
        grade=grade,
        visibility=visibility,
        status=status,
        limit=limit,
    )
    payload = {"count": len(records), "records": records}
    text = json.dumps(payload, indent=2)
    typer.echo(text)
    if output_json is not None:
        output_json = prepare_output_path(output_json)
        output_json.write_text(text, encoding="utf-8")
        typer.echo(f"Saved list: {output_json}")


@registry_app.command("export-public")
def registry_export_public(
    registry: Path = typer.Option(Path("registry/mdmp_registry.json"), help="Registry JSON path"),
    output_json: Path = typer.Option(Path("registry/public_bundle.json"), help="Output public bundle JSON"),
    privkey: Path | None = typer.Option(None, help="Optional signing key PEM for a signed public bundle"),
    signed_by: str = typer.Option("MDMP-Authority-v1", help="Signer identity label"),
    key_id: str = typer.Option("mdmp_pub_v1", help="Public key identifier"),
    expires_days: int | None = typer.Option(None, help="Optional signed bundle expiry in days"),
    passphrase: str | None = typer.Option(None, help="Private key passphrase", hide_input=True),
    passphrase_env: str | None = typer.Option(None, help="Environment variable containing the passphrase"),
    passphrase_file: Path | None = typer.Option(None, help="File containing the passphrase"),
) -> None:
    registry = safe_output_path(registry)
    output_json = safe_output_path(output_json)
    if privkey is not None:
        payload = export_signed_public_bundle(
            registry,
            output_json,
            signer=MDMPSigner(
                privkey,
                signed_by=signed_by,
                key_id=key_id,
                private_key_passphrase=resolve_passphrase(
                    passphrase=passphrase,
                    passphrase_env=passphrase_env,
                    passphrase_file=passphrase_file,
                ),
            ),
            expires_days=expires_days,
        )
    else:
        payload = export_public_bundle(registry, output_json)
    typer.echo(f"Exported public bundle: {output_json}")
    typer.echo(f"Public records: {len(payload.get('records', {}))}")


@registry_app.command("import-public")
def registry_import_public(
    bundle_json: Path,
    registry: Path = typer.Option(Path("registry/mdmp_registry.json"), help="Registry JSON path"),
    source: str = typer.Option("bundle", help="Source label for imported records"),
) -> None:
    registry = safe_output_path(registry)
    payload = import_public_bundle(registry, bundle_json, source=source)
    typer.echo(f"Imported records: {payload['imported']}")
    typer.echo(f"Registry: {payload['registry']}")


@registry_app.command("sync-url")
def registry_sync_url(
    url: str = typer.Option(..., help="Public bundle URL (JSON)"),
    registry: Path = typer.Option(Path("registry/mdmp_registry.json"), help="Registry JSON path"),
    source: str = typer.Option("remote", help="Source label for imported records"),
    public_key: Path | None = typer.Option(None, help="Optional public key PEM for signed bundle verification"),
    trust_store: Path | None = typer.Option(None, help="Optional trust store JSON for bundle verification"),
) -> None:
    registry = safe_output_path(registry)
    payload = sync_public_bundle_from_url(
        registry,
        url,
        source=source,
        public_key_path=public_key,
        trust_store_path=trust_store,
    )
    typer.echo(f"Synced records: {payload['imported']}")
    typer.echo(f"Registry: {payload['registry']}")


@integrations_app.command("dvc-stage")
def integration_dvc_stage(
    contract_path: str = typer.Option(..., help="Contract path used in mdmp validate"),
    dataset_path: str = typer.Option(..., help="Dataset path used in mdmp validate"),
    report_path: str = typer.Option("results/mdmp_report.json", help="Validation output report path"),
    stage_name: str = typer.Option("mdmp_validate", help="DVC stage name"),
    output_yaml: Path = typer.Option(Path("dvc.yaml"), help="Output dvc.yaml path"),
) -> None:
    output_yaml = safe_output_path(output_yaml)
    payload = build_dvc_stage(
        stage_name=stage_name,
        contract_path=contract_path,
        dataset_path=dataset_path,
        report_path=report_path,
    )
    write_dvc_stage(output_yaml, payload)
    typer.echo(f"Saved DVC stage: {output_yaml}")


@integrations_app.command("mlflow-log")
def integration_mlflow_log(
    artifact_path: Path = typer.Option(..., help="Artifact file to log"),
    target_path: str = typer.Option("mdmp", help="MLflow artifact path"),
) -> None:
    status = log_mdmp_artifact_to_mlflow(artifact_path, artifact_path=target_path)
    typer.echo(f"MLflow status: {status}")


@integrations_app.command("wandb-log")
def integration_wandb_log(
    artifact_path: Path = typer.Option(..., help="Artifact file to log"),
    name: str = typer.Option("mdmp-artifact", help="W&B artifact name"),
) -> None:
    status = log_mdmp_artifact_to_wandb(artifact_path, name=name)
    typer.echo(f"W&B status: {status}")


@app.command("hf-export")
def hf_export(
    dataset_id: str = typer.Option(..., help="Hugging Face dataset id (e.g. owner/name)"),
    report_json: Path | None = typer.Option(None, help="Optional MDMP report JSON"),
    fingerprint: str | None = typer.Option(None, help="Optional dataset fingerprint (sha256:...)"),
    grade: str | None = typer.Option(None, help="Optional grade override"),
    output_md: Path = typer.Option(Path("results/mdmp_hf_section.md"), help="Output Markdown path"),
    registry_url: str | None = typer.Option(None, help="Optional registry URL"),
) -> None:
    report = load_report(report_json) if report_json is not None else {}

    resolved_grade = grade or str(report.get("grade", report.get("mdmp_grade", "draft")))
    if fingerprint:
        resolved_fp = fingerprint
    else:
        fp_raw = report.get("dataset_fingerprint_sha256")
        if not fp_raw:
            raise typer.BadParameter("Provide --fingerprint or --report-json with dataset_fingerprint_sha256")
        resolved_fp = f"sha256:{fp_raw}" if not str(fp_raw).startswith("sha256:") else str(fp_raw)

    contract_fp_raw = report.get("contract_fingerprint_sha256")
    contract_fp = None
    if contract_fp_raw:
        contract_fp = (
            f"sha256:{contract_fp_raw}" if not str(contract_fp_raw).startswith("sha256:") else str(contract_fp_raw)
        )

    score = report.get("compliance_score")
    protocol = report.get("protocol_version", report.get("mdmp_protocol_version"))
    section = build_hf_mdmp_section(
        dataset_id=dataset_id,
        grade=resolved_grade,
        fingerprint=resolved_fp,
        contract_fingerprint=contract_fp,
        compliance_score=float(score) if score is not None else None,
        protocol_version=str(protocol) if protocol is not None else None,
        registry_url=registry_url,
    )
    output_md = prepare_output_path(output_md)
    output_md.write_text(section, encoding="utf-8")
    typer.echo(f"Saved HF MDMP section: {output_md}")


@app.command("conformance")
def conformance(
    workdir: Path = typer.Option(Path("results/conformance"), help="Working directory for suite artifacts"),
    output_json: Path = typer.Option(Path("results/mdmp_conformance.json"), help="Output conformance JSON"),
    strict: bool = typer.Option(
        True,
        "--strict/--no-strict",
        help="Exit non-zero when the conformance suite fails",
    ),
) -> None:
    workdir = safe_output_path(workdir)
    output_json = safe_output_path(output_json)
    payload = run_conformance_suite(workdir)
    write_conformance_report(output_json, payload)
    typer.echo(f"Saved conformance report: {output_json}")
    summary = payload.get("summary", {})
    typer.echo(f"Checks: {summary.get('passed')}/{summary.get('total')} passed")
    if strict and not payload.get("passed"):
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
