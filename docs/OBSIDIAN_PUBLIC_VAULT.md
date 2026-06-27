# Public Obsidian Knowledge Vault

The IINTS SDK also has a public Obsidian vault for people who want to explore the project as a connected knowledge base instead of a linear documentation site.

It contains:

- SDK user workflows
- hardware and edge notes
- Raspberry Pi Pico pump lab safety notes
- source and evidence library
- demo and presentation kits
- glossary and command maps
- mirrored official documentation for offline reading

> **Safety boundary:** the vault is for research, education, and documentation. It is not treatment guidance, not a medical device, and not a real insulin dosing workflow.

## Download

[Download the public Obsidian vault ZIP](assets/vaults/IINTS_SDK_Public_Obsidian_Vault.zip)

After downloading:

1. Unzip the file.
2. Open Obsidian.
3. Choose **Open folder as vault**.
4. Select `IINTS SDK Public Vault`.
5. Start at `00 START HERE.md`.

## Repository Hygiene

The repository keeps the public vault as a downloadable ZIP only. The expanded
Obsidian working folder is intentionally not tracked in Git because it duplicates
large parts of the documentation tree and includes editor-specific workspace
state.

## Obsidian Publish

This public vault is prepared as an Obsidian Publish package. Publishing still requires an Obsidian Publish account.

If you publish it:

- unzip `docs/assets/vaults/IINTS_SDK_Public_Obsidian_Vault.zip`
- publish the extracted folder `IINTS SDK Public Vault`
- do not publish the private/internal vault
- keep private notes, credentials, local results, and personal meeting notes out of the public vault

## Why This Exists

The docs site is best for quick navigation and formal documentation. The Obsidian vault is best for exploring how the SDK connects:

- commands link to workflows
- workflows link to sources
- sources link to claims
- claims link to demo scripts
- hardware notes link back to safety boundaries

That makes it easier for doctors, engineers, students, and jury members to inspect the project without needing to understand the whole codebase first.
