# Desktop App Signing

Unsigned desktop builds are useful for internal testing, but public Windows and macOS users will see security warnings until the app is signed with trusted certificates. This is normal platform behaviour, not an IINTS-AF-specific bug.

## What Removes The Warnings?

| Platform | Required for a trusted public build | Why |
| --- | --- | --- |
| Windows | Authenticode code-signing certificate, SHA-256 signature, timestamp | Identifies the publisher and helps Microsoft Defender SmartScreen build reputation. |
| macOS | Apple Developer Program membership, Developer ID Application certificate, hardened runtime signing, notarization, stapling | Lets Gatekeeper verify the app was signed by a trusted developer and accepted by Apple's notary service. |
| Linux | Usually no central signing warning for a direct executable | Linux distributions vary; AppImage/deb/rpm signing can be added later for distro-style trust. |

## GitHub Actions Support

The `Desktop Beta Builds` workflow already contains optional signing steps. If the secrets below are not configured, the workflow skips signing and still publishes unsigned beta artifacts.

### Windows Secrets

Add these repository secrets:

| Secret | Meaning |
| --- | --- |
| `WINDOWS_SIGNING_PFX_BASE64` | Base64-encoded `.pfx` Authenticode code-signing certificate. |
| `WINDOWS_SIGNING_PFX_PASSWORD` | Password for the `.pfx` file. |

The workflow signs `dist/IINTS-AF-Desktop-Beta.exe` using `signtool`, SHA-256 file digest, and RFC 3161 timestamping before packaging the release asset.

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

## Important Notes

- Signing reduces warnings, but Windows SmartScreen may still warn for a new publisher until enough reputation is built.
- EV code-signing certificates can build SmartScreen trust faster, but they cost more and usually require stricter identity validation.
- Never commit certificates, `.pfx`, `.p12`, passwords, or Apple credentials to the repository.
- Keep unsigned beta builds clearly labelled as beta/research builds.
- IINTS-AF remains research and education software only; signing means the binary identity is trusted, not that the software is a medical device.
