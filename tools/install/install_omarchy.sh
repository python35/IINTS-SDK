#!/usr/bin/env bash
set -euo pipefail

readonly DEFAULT_PYTHON_VERSION="3.11"
readonly SDK_PACKAGE="iints-sdk-python35"
readonly RELEASE_BASE_URL="https://github.com/python35/IINTS-SDK/releases/download/tauri-beta-latest"
readonly APPIMAGE_NAME="IINTS-AF-Research-Workbench-linux-x64.AppImage"
readonly ICON_URL="https://raw.githubusercontent.com/python35/IINTS-SDK/tauri-beta-latest/src/iints/assets/iints_logo.png"

profile="standard"
python_version="$DEFAULT_PYTHON_VERSION"
sdk_version=""
install_desktop=false
dry_run=false

usage() {
  cat <<'EOF'
Install IINTS-AF on Omarchy Linux using Omarchy's package tools and Mise.

Usage:
  install_omarchy.sh [options]

Options:
  --profile standard|research|desktop
                            standard: simulation, reports, and MDMP (default)
                            research: standard plus AI/data research libraries
                            desktop: complete Python engine plus the AppImage
  --desktop               also install the native Research Workbench AppImage
  --python-version VERSION
                          Mise Python version; must be 3.10-3.14 (default: 3.11)
  --version VERSION       pin the Python SDK to an exact release
  --dry-run               print every planned action without changing the system
  -h, --help              show this help

The installer writes only to ~/.iints-af and ~/.local, apart from packages added
through `omarchy pkg add`. It never runs pacman or yay directly.
EOF
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

run_command() {
  if "$dry_run"; then
    print_command "$@"
  else
    "$@"
  fi
}

while (($#)); do
  case "$1" in
    --profile)
      (($# >= 2)) || fail "--profile requires a value"
      profile="$2"
      shift 2
      ;;
    --desktop)
      install_desktop=true
      shift
      ;;
    --python-version)
      (($# >= 2)) || fail "--python-version requires a value"
      python_version="$2"
      shift 2
      ;;
    --version)
      (($# >= 2)) || fail "--version requires a value"
      sdk_version="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "Unknown option: $1"
      ;;
  esac
done

case "$profile" in
  standard)
    extras="full,mdmp"
    ;;
  research)
    extras="full,mdmp,research"
    ;;
  desktop)
    extras="desktop-all"
    install_desktop=true
    ;;
  *)
    fail "Unknown profile '$profile'; use standard, research, or desktop"
    ;;
esac

[[ "$python_version" =~ ^3\.(10|11|12|13|14)(\.[0-9]+)?$ ]] || \
  fail "Python must be in the SDK-supported 3.10-3.14 range"
[[ -z "$sdk_version" || "$sdk_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([a-zA-Z0-9.-]+)?$ ]] || \
  fail "--version must look like 1.5.34"

if "$install_desktop" && [[ "$(uname -m)" != "x86_64" ]] && ! "$dry_run"; then
  fail "The current Linux AppImage is x86_64-only. Use the CLI profile on this architecture."
fi

if ! "$dry_run"; then
  [[ "$(uname -s)" == "Linux" ]] || fail "This installer is only for Omarchy Linux"
  if ! command -v omarchy >/dev/null 2>&1 && ! command -v omarchy-pkg-add >/dev/null 2>&1; then
    fail "Omarchy package tooling was not found. Use docs/OMARCHY_INSTALL.md for manual installation."
  fi
fi

package_spec="${SDK_PACKAGE}[${extras}]"
if [[ -n "$sdk_version" ]]; then
  package_spec+="==${sdk_version}"
fi

readonly state_dir="${HOME}/.iints-af"
readonly engine_dir="${state_dir}/python-engine"
readonly local_bin="${HOME}/.local/bin"
readonly app_dir="${HOME}/.local/opt/iints-af"
readonly applications_dir="${HOME}/.local/share/applications"
readonly icon_dir="${HOME}/.local/share/icons/hicolor/256x256/apps"

printf 'IINTS-AF Omarchy installer\n'
printf '  Profile: %s\n' "$profile"
printf '  Python:  %s (managed by Mise)\n' "$python_version"
printf '  SDK:     %s\n' "$package_spec"
printf '  Desktop: %s\n' "$install_desktop"
if "$dry_run"; then
  printf '  Mode:    dry run; no files or packages will be changed\n'
fi

install_omarchy_package() {
  local package="$1"
  if "$dry_run"; then
    print_command omarchy pkg add "$package"
  elif command -v omarchy >/dev/null 2>&1; then
    omarchy pkg add "$package"
  else
    omarchy-pkg-add "$package"
  fi
}

printf '\n[1/5] Preparing Omarchy packages\n'
install_omarchy_package mise
install_omarchy_package curl
if "$install_desktop"; then
  install_omarchy_package fuse2
fi

printf '\n[2/5] Preparing Python %s with Mise\n' "$python_version"
run_command mise use --global "python@${python_version}"

create_engine() {
  local staging_dir="${state_dir}/python-engine.new.$$"
  local backup_dir="${state_dir}/python-engine.previous.$(date +%Y%m%d%H%M%S)"

  run_command mkdir -p "$state_dir"
  if "$dry_run"; then
    print_command mise exec "python@${python_version}" -- python -m venv "$engine_dir"
    return
  fi

  rm -rf "$staging_dir"
  mise exec "python@${python_version}" -- python -m venv "$staging_dir"
  if [[ -d "$engine_dir" ]]; then
    mv "$engine_dir" "$backup_dir"
    printf 'Previous Python engine retained at %s\n' "$backup_dir"
  fi
  mv "$staging_dir" "$engine_dir"
}

printf '\n[3/5] Installing the private IINTS-AF Python engine\n'
if [[ -x "${engine_dir}/bin/python" ]] && ! "$dry_run"; then
  installed_python="$(${engine_dir}/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  requested_python="${python_version%.*}"
  if [[ "$python_version" =~ ^3\.[0-9]+$ ]]; then
    requested_python="$python_version"
  fi
  if [[ "$installed_python" != "$requested_python" ]]; then
    printf 'Replacing Python %s engine with Python %s.\n' "$installed_python" "$python_version"
    create_engine
  else
    printf 'Reusing compatible engine at %s\n' "$engine_dir"
  fi
else
  create_engine
fi

run_command "${engine_dir}/bin/python" -m pip install --upgrade pip
run_command "${engine_dir}/bin/python" -m pip install --upgrade "$package_spec"
run_command mkdir -p "$local_bin"

if "$dry_run"; then
  print_command ln -sfn "${engine_dir}/bin/iints" "${local_bin}/iints"
elif [[ -e "${local_bin}/iints" && ! -L "${local_bin}/iints" ]]; then
  printf 'WARNING: %s is a regular file and was not replaced.\n' "${local_bin}/iints" >&2
else
  ln -sfn "${engine_dir}/bin/iints" "${local_bin}/iints"
fi

install_workbench() (
  local asset_url="${RELEASE_BASE_URL}/${APPIMAGE_NAME}"
  local checksum_url="${asset_url}.sha256"
  local appimage_path="${app_dir}/${APPIMAGE_NAME}"
  local launcher_path="${local_bin}/iints-workbench"
  local icon_path="${icon_dir}/iints-af.png"
  local desktop_path="${applications_dir}/org.iints.research-workbench.desktop"

  printf '\n[4/5] Installing the native Research Workbench\n'
  if "$dry_run"; then
    print_command curl --fail --location "$asset_url"
    print_command curl --fail --location "$checksum_url"
    print_command sha256sum --check "${APPIMAGE_NAME}.sha256"
    printf '  install AppImage -> %s\n' "$appimage_path"
    printf '  install launcher -> %s\n' "$launcher_path"
    printf '  install desktop entry -> %s\n' "$desktop_path"
    return
  fi

  local staging_dir
  staging_dir="$(mktemp -d)"
  trap 'rm -rf "$staging_dir"' EXIT
  curl --fail --location --retry 3 --output "${staging_dir}/${APPIMAGE_NAME}" "$asset_url"
  curl --fail --location --retry 3 --output "${staging_dir}/${APPIMAGE_NAME}.sha256" "$checksum_url"
  (
    cd "$staging_dir"
    sha256sum --check "${APPIMAGE_NAME}.sha256"
  )

  install -Dm755 "${staging_dir}/${APPIMAGE_NAME}" "$appimage_path"
  install -d "$local_bin" "$applications_dir" "$icon_dir"
  curl --fail --location --retry 3 --output "$icon_path" "$ICON_URL"

  cat >"$launcher_path" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export IINTS_PYTHON="${engine_dir}/bin/python"
appimage="${appimage_path}"
if [[ -c /dev/fuse ]]; then
  exec "\$appimage" "\$@"
fi
exec "\$appimage" --appimage-extract-and-run "\$@"
EOF
  chmod 0755 "$launcher_path"

  cat >"$desktop_path" <<EOF
[Desktop Entry]
Type=Application
Name=IINTS-AF Research Workbench
Comment=Open-source diabetes technology research workbench
Exec=${launcher_path}
Icon=${icon_path}
Terminal=false
Categories=Education;Science;Development;
StartupNotify=true
EOF

  if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$applications_dir" >/dev/null || true
  fi
)

if "$install_desktop"; then
  install_workbench
else
  printf '\n[4/5] Desktop AppImage not requested\n'
fi

printf '\n[5/5] Verifying the SDK\n'
run_command "${engine_dir}/bin/iints" --version
run_command "${engine_dir}/bin/iints" doctor --smoke-run --suggest

workbench_path="not installed"
if "$install_desktop"; then
  workbench_path="${local_bin}/iints-workbench"
fi

cat <<EOF

Installation plan completed.
CLI:      ${local_bin}/iints
Engine:   ${engine_dir}
Workbench: ${workbench_path}

If the commands are not found in a new terminal, add ~/.local/bin to PATH.
For reproducible studies, rerun with --version X.Y.Z to pin an SDK release.
EOF
