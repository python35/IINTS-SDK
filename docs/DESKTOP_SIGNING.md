# Desktop App Signing

Unsigned desktop builds are useful for internal testing, but public Windows and macOS users will see security warnings until the app is signed with trusted certificates. This is normal platform behaviour, not an IINTS-AF-specific bug.

## What Removes The Warnings?

| Platform | Required for a trusted public build | Why |
| --- | --- | --- |
| Windows | Authenticode code-signing certificate, SHA-256 signature, timestamp | Identifies the publisher and helps Microsoft Defender SmartScreen build reputation. |
| macOS | Apple Developer Program membership, Developer ID Application certificate, hardened runtime signing, notarization, stapling | Lets Gatekeeper verify the app was signed by a trusted developer and accepted by Apple's notary service. |
| Linux | Usually no central signing warning for a direct executable | Linux distributions vary; AppImage/deb/rpm signing can be added later for distro-style trust. |

## GitHub Actions Support

The `Rust Desktop Beta Builds` workflow signs with a real certificate when the secrets below are configured, and otherwise falls back to an ad-hoc macOS signature / unsigned Windows installer / skipped notarization for every build, including public tag-triggered releases. This applies uniformly regardless of `MACOS_CERTIFICATE_P12_BASE64` / `WINDOWS_SIGNING_PFX_BASE64` / notarization credentials being present, so a `tauri-beta-v*` release still publishes without any certificates configured -- end users will just see the platform warnings described above until real certificates are added.

### Windows Secrets

Add these repository secrets:

| Secret | Meaning |
| --- | --- |
| `WINDOWS_SIGNING_PFX_BASE64` | Base64-encoded `.pfx` Authenticode code-signing certificate. |
| `WINDOWS_SIGNING_PFX_PASSWORD` | Password for the `.pfx` file. |

The workflow signs the generated Tauri NSIS installer using `signtool`, SHA-256 file digest, and RFC 3161 timestamping before packaging the release asset.

### macOS Secrets

Add these repository secrets:

| Secret | Meaning |
| --- | --- |
| `MACOS_CERTIFICATE_P12_BASE64` | Base64-encoded exported Developer ID Application `.p12` certificate. |
| `MACOS_CERTIFICATE_PASSWORD` | Password for the `.p12` certificate. |
| `MACOS_SIGNING_IDENTITY` | Exact signing identity, for example `Developer ID Application: Name (TEAMID)`. |
| `MACOS_KEYCHAIN_PASSWORD` | Repository secret used only to lock/unlock the temporary CI keychain. Required when macOS signing is enabled. |
| `APPLE_ID` | Apple Developer account email used for notarization. |
| `APPLE_TEAM_ID` | Apple Developer Team ID. |
| `APPLE_APP_SPECIFIC_PASSWORD` | App-specific password for notarization. |

The workflow imports the certificate into a temporary keychain, signs the `.app` with hardened runtime, creates the `.dmg`, submits it to Apple's notary service with `xcrun notarytool`, staples the notarization ticket, validates it, and rewrites the `.sha256` checksum.

## Updater Signing

The app can check for and install its own updates from inside Settings, instead of the user
re-downloading an installer manually. This uses the Tauri updater plugin, which is signed with its
own lightweight keypair -- unrelated to the platform code-signing certificates above, and required
either way (there is no unsigned fallback for updates, since an unsigned auto-updater would let
anyone who could intercept or spoof the update feed run arbitrary code on every installed copy).

| Secret | Meaning |
| --- | --- |
| `TAURI_SIGNING_PRIVATE_KEY` | The updater's minisign private key, generated once with `npx tauri signer generate` (already configured for this repository). |
| `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` | Password protecting that private key. |

The matching public key is committed in `apps/iints-tauri/src-tauri/tauri.conf.json` under
`plugins.updater.pubkey`. When both secrets are present, `tauri build` additionally produces a
signed updater artifact (`.app.tar.gz` on macOS, the installer itself on Windows and Linux) next to
the normal installer, and the release workflow assembles those into `latest.json`, published at the
stable `tauri-beta-latest` release alongside the installers themselves. The app's updater endpoint
in `tauri.conf.json` always reads that one file, so `tauri-beta-latest` is the only URL it depends
on; per-version tagged releases keep their own copies of the same generated manifest for the record,
but nothing reads them.

**If this key is ever lost**, generate a new one and replace `plugins.updater.pubkey`, but understand
the consequence first: every already-installed copy of the app only trusts the *old* public key, so
none of them will accept updates signed with the new key. Existing users would need to reinstall
from a manually downloaded installer at least once to pick up the new key, after which their
in-app updater works again.

## Important Notes

- Signing reduces warnings, but Windows SmartScreen may still warn for a new publisher until enough reputation is built.
- EV code-signing certificates can build SmartScreen trust faster, but they cost more and usually require stricter identity validation.
- Never commit certificates, `.pfx`, `.p12`, passwords, or Apple credentials to the repository.
- Keep unsigned developer artifacts private to CI testing; do not redistribute them as official downloads.
- IINTS-AF remains research and education software only; signing means the binary identity is trusted, not that the software is a medical device.
