#!/usr/bin/env bash

set -euo pipefail

version="${1:-}"
if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "usage: Scripts/package_app.sh VERSION [OUTPUT_DIRECTORY]" >&2
  exit 64
fi

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "$script_directory/.." && pwd)"
compiled_version="$(ruby -e 'puts File.read(ARGV[0])[/static let current\s*=\s*"([^"]+)"/, 1]' "$repository_root/Sources/TUFFModelCatalog/TUFFVersion.swift")"
if [[ "$version" != "$compiled_version" ]]; then
  echo "package version $version disagrees with TUFFVersion.current ($compiled_version)" >&2
  exit 64
fi
bundle_identifier="${TUFF_BUNDLE_IDENTIFIER:-com.rexmhall09.TUFF}"
if [[ ! "$bundle_identifier" =~ ^[A-Za-z0-9]+(\.[A-Za-z0-9-]+)+$ ]]; then
  echo "invalid bundle identifier" >&2
  exit 64
fi
output_directory="${2:-$repository_root/dist}"
mkdir -p "$output_directory"
output_directory="$(cd -- "$output_directory" && pwd)"

app="$output_directory/TUFF.app"
archive_name="TUFF-v${version}-macos-arm64.zip"
archive="$output_directory/$archive_name"
checksum="$archive.sha256"
update_feed_url="${TUFF_UPDATE_FEED_URL:-https://github.com/tuff-app/TUFF/releases/latest/download/appcast.xml}"
update_public_key="${TUFF_UPDATE_PUBLIC_ED_KEY:-}"
update_public_key_file="$repository_root/Config/UpdateSigningPublicKey.txt"
automatic_updates=true
if [[ "${TUFF_LOCAL_BUILD:-0}" == 1 ]]; then automatic_updates=false; fi
if [[ -z "$update_public_key" && -f "$update_public_key_file" ]]; then
  IFS= read -r update_public_key < "$update_public_key_file"
fi

rm -rf "$app"
rm -f "$archive" "$checksum"

temporary_directory="$(mktemp -d "${TMPDIR:-/tmp}/tuff-package.XXXXXX")"
trap 'rm -rf "$temporary_directory"' EXIT

cd "$repository_root"
ruby Scripts/check_brand_assets.rb
# This package only consumes the local package graph and its checked-out
# dependencies. Disabling SwiftPM's command sandbox keeps release packaging
# usable in restricted developer shells where the manifest sandbox cannot
# create its compiler module cache.
#
# The link has to be told the SDK with `-isysroot`. SwiftPM's Swift Build
# backend (Swift 6.4) links through clang with `--sysroot` only, and clang
# then records the deployment target as the SDK version: 5.0.2 shipped
# stamped "SDK 15.0", so macOS ran it with the pre-Tahoe look instead of
# Liquid Glass.
sdk_path="$(xcrun --sdk macosx --show-sdk-path)"
build_flags=(
  -c release --disable-sandbox
  -Xswiftc -Xclang-linker -Xswiftc -isysroot
  -Xswiftc -Xclang-linker -Xswiftc "$sdk_path"
)
swift build "${build_flags[@]}"
binary_directory="$(swift build "${build_flags[@]}" --show-bin-path)"
linked_sdk="$(otool -l "$binary_directory/TUFF" \
  | awk '/LC_BUILD_VERSION/ { found = 1 } found && $1 == "sdk" && !printed { print $2; printed = 1 }')"
if [[ "${linked_sdk%%.*}" -lt 26 ]]; then
  echo "TUFF was linked against SDK ${linked_sdk:-unknown}; macOS would show the legacy appearance" >&2
  exit 1
fi

required_binaries=(
  TUFF
  TUFFCommand
  TUFFCLI
  TUFFRepack
  TUFFServer
)
required_bundles=(
  TUFF_TUFFAppCore.bundle
  TUFF_TUFFMac.bundle
  swift-nio_NIOPosix.bundle
  SwiftMath_SwiftMath.bundle
)
required_frameworks=(Sparkle.framework)

for name in "${required_binaries[@]}" "${required_bundles[@]}" \
  "${required_frameworks[@]}"; do
  if [[ ! -e "$binary_directory/$name" ]]; then
    echo "missing release product: $binary_directory/$name" >&2
    exit 1
  fi
done

mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources" \
  "$app/Contents/Frameworks"
signed_binaries="$temporary_directory/signed-binaries"
mkdir -p "$signed_binaries"
# App, CLI and HTTP host roles share the small host executable. Inference
# helpers and shaders live in authenticated packages outside this bundle.
shipped_binaries=(TUFF TUFFCommand TUFFRepack)
for name in "${shipped_binaries[@]}"; do
  install -m 0755 "$binary_directory/$name" "$signed_binaries/$name"
done

# Drop local symbols before signing. `-x` keeps external symbols, so dynamic
# lookup and Sparkle's framework linkage are untouched; what is lost is
# function names in a crash report, which an ad-hoc-signed build has no
# symbol server for anyway. Stripping must happen before codesign — doing it
# after invalidates the signature.
# Keep the unstripped copies next to the archive. Symbolicating a crash from
# a shipped build needs a binary whose UUID matches, and `.build` is rebuilt
# constantly — the first crash report after stripping was unreadable until the
# matching binary turned out to still be lying around by luck.
symbols_directory="$output_directory/TUFF-v${version}-symbols"
rm -rf "$symbols_directory"
mkdir -p "$symbols_directory"
for name in "${shipped_binaries[@]}"; do
  install -m 0644 "$signed_binaries/$name" "$symbols_directory/$name"
  strip -x "$signed_binaries/$name"
done
if ! otool -l "$signed_binaries/TUFF" \
  | grep -F '@executable_path/../Frameworks' >/dev/null; then
  install_name_tool -add_rpath '@executable_path/../Frameworks' \
    "$signed_binaries/TUFF"
fi
for name in "${shipped_binaries[@]}"; do
  codesign --force --sign - --timestamp=none "$signed_binaries/$name"
  codesign --verify --strict --verbose=2 "$signed_binaries/$name"
done
install -m 0755 "$signed_binaries/TUFF" "$app/Contents/MacOS/TUFF"
mkdir -p "$app/Contents/Resources/bin"
install -m 0755 "$signed_binaries/TUFFCommand" \
  "$app/Contents/Resources/bin/tuff"
install -m 0755 "$signed_binaries/TUFFRepack" \
  "$app/Contents/Resources/bin/TUFFRepack"
for name in TUFFCLI TUFFServer; do
  ln -s ../../MacOS/TUFF "$app/Contents/Resources/bin/$name"
done

# Keep resource bundles in the standard sealed app resource directory. TUFF's
# resource lookups prefer this location in a packaged app and fall back to
# SwiftPM's generated Bundle.module path for command-line and test builds.
for name in "${required_bundles[@]}"; do
  ditto "$binary_directory/$name" "$app/Contents/Resources/$name"
done

engine_catalog="${TUFF_ENGINE_CATALOG_PATH:-}"
if [[ -z "$engine_catalog" || ! -f "$engine_catalog" ]]; then
  echo "Set TUFF_ENGINE_CATALOG_PATH to a verified engine catalogue (Scripts/package_engines.sh builds local packages)." >&2
  exit 1
fi
install -m 0644 "$engine_catalog" "$app/Contents/Resources/engine-catalog.json"
if nm -gU "$signed_binaries/TUFF" | grep -E 'RealForwardRunner|GPTOSSForwardRunner|MetalContext' >/dev/null; then
  echo "host unexpectedly includes an inference runtime" >&2
  exit 1
fi

# Vendor/SwiftMath carries only Latin Modern Math, the default the renderer
# uses; it never sets `MathImage.font`. SwiftMath loads the font by name the
# first time a formula is drawn and stops the app if the file is missing, so
# check for it here rather than in someone's chat.
math_resources="$app/Contents/Resources/SwiftMath_SwiftMath.bundle"
if [[ -d "$math_resources/Contents/Resources" ]]; then
  math_resources="$math_resources/Contents/Resources"
fi
math_fonts="$math_resources/mathFonts.bundle"
for extension in otf plist; do
  if [[ ! -s "$math_fonts/latinmodern-math.$extension" ]]; then
    echo "SwiftMath is missing latinmodern-math.$extension" >&2
    exit 1
  fi
done
for name in "${required_frameworks[@]}"; do
  ditto "$binary_directory/$name" "$app/Contents/Frameworks/$name"
done

install -m 0644 LICENSE "$app/Contents/Resources/LICENSE"
install -m 0644 README.md "$app/Contents/Resources/README.md"
install -m 0644 THIRD_PARTY_NOTICES.md \
  "$app/Contents/Resources/THIRD_PARTY_NOTICES.md"

# Compile the Icon Composer source the way Xcode does. actool emits the layered
# Liquid Glass icon into Assets.car for macOS 26 and a flattened TUFF.icns
# alongside it, which older systems fall back to through CFBundleIconFile.
icon_build="$temporary_directory/icon"
mkdir -p "$icon_build"
icon_partial_plist="$temporary_directory/icon-partial.plist"
xcrun actool \
  --compile "$icon_build" \
  --platform macosx \
  --minimum-deployment-target 15.0 \
  --app-icon TUFF \
  --output-partial-info-plist "$icon_partial_plist" \
  "$repository_root/Brand/TUFF.icon" >/dev/null

for produced in Assets.car TUFF.icns; do
  if [[ ! -s "$icon_build/$produced" ]]; then
    echo "actool did not produce $produced" >&2
    exit 1
  fi
  install -m 0644 "$icon_build/$produced" "$app/Contents/Resources/$produced"
done

icon_name="$(plutil -extract CFBundleIconName raw -o - "$icon_partial_plist")"
icon_file="$(plutil -extract CFBundleIconFile raw -o - "$icon_partial_plist")"
if [[ -z "$icon_name" || -z "$icon_file" ]]; then
  echo "actool did not report the icon Info.plist keys" >&2
  exit 1
fi

info_plist="$app/Contents/Info.plist"
plutil -create xml1 "$info_plist"
plutil -insert CFBundleDevelopmentRegion -string en "$info_plist"
plutil -insert CFBundleDisplayName -string TUFF "$info_plist"
plutil -insert CFBundleExecutable -string TUFF "$info_plist"
plutil -insert CFBundleIconFile -string "$icon_file" "$info_plist"
plutil -insert CFBundleIconName -string "$icon_name" "$info_plist"
plutil -insert CFBundleIdentifier -string "$bundle_identifier" "$info_plist"
plutil -insert CFBundleInfoDictionaryVersion -string 6.0 "$info_plist"
plutil -insert CFBundleName -string TUFF "$info_plist"
plutil -insert CFBundlePackageType -string APPL "$info_plist"
plutil -insert CFBundleShortVersionString -string "$version" "$info_plist"
plutil -insert CFBundleVersion -string "$version" "$info_plist"
plutil -insert LSApplicationCategoryType -string public.app-category.utilities \
  "$info_plist"
plutil -insert LSMinimumSystemVersion -string 15.0 "$info_plist"
plutil -insert LSRequiresNativeExecution -bool true "$info_plist"
plutil -insert NSHighResolutionCapable -bool true "$info_plist"
if [[ -n "$update_public_key" ]]; then
  if ! ruby -rbase64 -e \
    'decoded = Base64.strict_decode64(ARGV.fetch(0)); exit(decoded.bytesize == 32 ? 0 : 1)' \
    "$update_public_key" 2>/dev/null; then
    echo "invalid Sparkle EdDSA public key" >&2
    exit 1
  fi
  if [[ "$update_feed_url" != https://* ]]; then
    echo "update feed must use HTTPS" >&2
    exit 1
  fi
  plutil -insert SUFeedURL -string "$update_feed_url" "$info_plist"
  plutil -insert SUPublicEDKey -string "$update_public_key" "$info_plist"
  plutil -insert SURequireSignedFeed -bool true "$info_plist"
  plutil -insert SUVerifyUpdateBeforeExtraction -bool true "$info_plist"
  plutil -insert SUEnableAutomaticChecks -bool "$automatic_updates" "$info_plist"
  plutil -insert SUAllowsAutomaticUpdates -bool true "$info_plist"
  plutil -insert SUAutomaticallyUpdate -bool "$automatic_updates" "$info_plist"
  if [[ "$automatic_updates" == false ]]; then plutil -insert TUFFLocalBuild -bool true "$info_plist"; fi
else
  echo "warning: no Sparkle update public key; in-app updates are disabled" >&2
fi

if [[ -f "$repository_root/Sources/TUFFServer/Core/RouterServerRuntime.swift" ]]; then
agent_directory="$app/Contents/Library/LaunchAgents"
mkdir -p "$agent_directory"
agent_plist="$agent_directory/$bundle_identifier.server.plist"
plutil -create xml1 "$agent_plist"
plutil -insert Label -string "$bundle_identifier.server" "$agent_plist"
plutil -insert BundleProgram -string Contents/Resources/bin/TUFFServer "$agent_plist"
plutil -insert ProgramArguments -json '["TUFFServer", "--background"]' "$agent_plist"
plutil -insert RunAtLoad -bool true "$agent_plist"
plutil -insert KeepAlive -json '{"SuccessfulExit":false}' "$agent_plist"
plutil -lint "$agent_plist"
fi
plutil -lint "$info_plist"
if [[ -n "$update_public_key" ]]; then
  [[ "$(plutil -extract SUEnableAutomaticChecks raw "$info_plist")" == "$automatic_updates" ]]
  [[ "$(plutil -extract SUAllowsAutomaticUpdates raw "$info_plist")" == true ]]
  [[ "$(plutil -extract SUAutomaticallyUpdate raw "$info_plist")" == "$automatic_updates" ]]
fi

main_architectures="$(lipo -archs "$app/Contents/MacOS/TUFF")"
if [[ "$main_architectures" != "arm64" ]]; then
  echo "release app must contain arm64-only executables" >&2
  exit 1
fi
for name in tuff TUFFCLI TUFFRepack TUFFServer; do
  if [[ "$(lipo -archs "$app/Contents/Resources/bin/$name")" != "arm64" ]]; then
    echo "release CLI must contain arm64-only executable: $name" >&2
    exit 1
  fi
done

# Sign after the complete bundle has been assembled so Info.plist and resources
# are sealed as part of the app rather than leaving a standalone executable
# signature inside an otherwise unsigned bundle.
codesign --force --deep --sign - --timestamp=none \
  "$app/Contents/Frameworks/Sparkle.framework"
codesign --force --deep --sign - --timestamp=none "$app"
codesign --verify --deep --strict --verbose=2 "$app"

ditto -c -k --sequesterRsrc --keepParent "$app" "$archive"
(
  cd "$output_directory"
  shasum -a 256 "$archive_name" > "$archive_name.sha256"
)

extracted="$temporary_directory/extracted"
mkdir -p "$extracted"
ditto -x -k "$archive" "$extracted"
codesign --verify --deep --strict --verbose=2 "$extracted/TUFF.app"
test -f "$extracted/TUFF.app/Contents/Frameworks/Sparkle.framework/Sparkle"
test -x "$extracted/TUFF.app/Contents/Resources/bin/tuff"
"$extracted/TUFF.app/Contents/Resources/bin/tuff" --help \
  | grep -F 'tuff prompt' >/dev/null
for link in Resources/bin/TUFFCLI Resources/bin/TUFFServer; do
  test -L "$extracted/TUFF.app/Contents/$link"
done
"$extracted/TUFF.app/Contents/Resources/bin/TUFFCLI" --help \
  | grep -F 'usage: TUFFCLI' >/dev/null
"$extracted/TUFF.app/Contents/Resources/bin/TUFFServer" --help \
  | grep -F 'usage: TUFFServer' >/dev/null
# Homebrew links `tuff` onto the PATH, where the shell passes it the bare
# name. Run it that way from elsewhere: it must still find TUFFCLI.
(cd / && PATH="$extracted/TUFF.app/Contents/Resources/bin:$PATH" \
  tuff prompt --help < /dev/null) | grep -F 'usage: TUFFCLI' >/dev/null
otool -L "$extracted/TUFF.app/Contents/MacOS/TUFF" \
  | grep -F '@rpath/Sparkle.framework/Versions/B/Sparkle' >/dev/null

echo "created $archive"
echo "created $checksum"
