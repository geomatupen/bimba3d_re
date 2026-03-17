# WiX Burn Starter for Bimba3D

This folder contains a starter template for a one-click Windows chainer installer.

## Files
- `Bimba3D.Bundle.wxs` - Burn bundle template with package chain placeholders.
- `EULA.md` - Bimba3D installer EULA text.
- `THIRD_PARTY_NOTICES.md` - third-party license and EULA references used by installer flow.
- `payloads/payload-manifest.json` - versioned URLs + SHA256 pins.
- `payloads/install-colmap.cmd` - extracts COLMAP zip and sets `COLMAP_EXE`.
- `scripts/download-payloads.ps1` - downloads payloads from manifest.
- `scripts/update-manifest-sha256.ps1` - computes SHA256 from local files and can update manifest.
- `scripts/validate-payload-manifest.ps1` - fails on placeholder URLs/SHA256.
- `scripts/build-bundle.ps1` - validates payloads and builds installer EXE.
- `scripts/sign-installer.ps1` - signs and verifies installer EXE with signtool.
- `scripts/release.ps1` - one-command wrapper for validate/build/sign.

## What to customize
1. Keep `LicenseUrl` pointed to your own EULA page/file and maintain the third-party notices file.
2. Set real SHA256 values in `payloads/payload-manifest.json`.
3. `Bimba3D.msi` URL is preconfigured for GitHub Releases latest asset (`geomatupen/bimba3d`).
  - Keep it as-is if your release asset name is exactly `Bimba3D.msi`.
  - Change only if you use a different file name or repo.
4. Decide whether `InstallBuildTools` should be 0 (runtime only) or 1 (include compile toolchain).
5. Decide whether `InstallCudaToolkit` should be 0 (default opt-in) or 1 (auto-install if missing).
5. Add code signing to final `Bimba3D-Setup.exe`.

Default compliance behavior in this repo:
- `VC++ Runtime` auto-installs if missing.
- `Build Tools` is opt-in (`InstallBuildTools=0` by default).
- `CUDA Toolkit` is opt-in (`InstallCudaToolkit=0` by default).
- Bundle license link points to project EULA: `installer/wix/EULA.md`.
- Third-party obligations are summarized in: `installer/wix/THIRD_PARTY_NOTICES.md`.

## Suggested build tooling
- Install WiX v4 (`wix.exe`).

## Build workflow
1. Download payloads:
  - `powershell -ExecutionPolicy Bypass -File .\scripts\download-payloads.ps1`
2. Place your app MSI in `payloads\Bimba3D.msi` (or set manifest URL and download it).
3. Validate manifest (required for release):
  - `powershell -ExecutionPolicy Bypass -File .\scripts\validate-payload-manifest.ps1 -RequireLocalFiles`
3. Build bundle:
  - `powershell -ExecutionPolicy Bypass -File .\scripts\build-bundle.ps1 -CiStrict`
4. Sign installer (thumbprint example):
  - `powershell -ExecutionPolicy Bypass -File .\scripts\sign-installer.ps1 -InstallerPath .\Bimba3D-Setup.exe -CertificateThumbprint <THUMBPRINT>`

PFX signing example:
- `powershell -ExecutionPolicy Bypass -File .\scripts\sign-installer.ps1 -InstallerPath .\Bimba3D-Setup.exe -PfxPath C:\certs\codesign.pfx -PfxPassword <PASSWORD>`

## One-command release wrapper
- Full run (downloads + hashes from local files + build + sign):
  - `powershell -ExecutionPolicy Bypass -File .\scripts\release.ps1 -DownloadPayloads -UpdateShaFromLocalFiles -CertificateThumbprint <THUMBPRINT>`
- Local-only build (no signing):
  - `powershell -ExecutionPolicy Bypass -File .\scripts\release.ps1 -UpdateShaFromLocalFiles -SkipSigning`

## Compliance launcher (interactive consent)
- Use this launcher to explicitly ask users before Build Tools / CUDA install and pass flags to Burn:
  - `powershell -ExecutionPolicy Bypass -File .\scripts\start-compliant-install.ps1 -InstallerPath .\Bimba3D-Setup.exe`
- Launcher prints links to:
  - Bimba3D EULA
  - Third-party notices
  - Build Tools terms
  - CUDA EULA

## What to replace in payload-manifest.json
1. `sha256` values:
   - Preferred: place payload files in `payloads\` then run:
     - `powershell -ExecutionPolicy Bypass -File .\scripts\update-manifest-sha256.ps1 -UpdateManifest`
   - This writes actual SHA256 values automatically.
2. `url` for `bimba3d_app_msi`:
  - Already set to: `https://github.com/geomatupen/bimba3d/releases/latest/download/Bimba3D.msi`
  - If hosting MSI elsewhere (or using a different asset name), set that direct HTTPS URL.
   - If not hosting and using local file only, set `"url": "LOCAL_FILE"` and place `Bimba3D.msi` in `payloads\`.
3. Keep vendor URLs for VS/VC++/CUDA/COLMAP unless you mirror internally.

No login is required to calculate SHA256. Login is only needed if your MSI is stored in a private artifact repository.

Direct WiX command used by the script:
- `wix build Bimba3D.Bundle.wxs -ext WixToolset.Bal.wixext -ext WixToolset.Util.wixext -o Bimba3D-Setup.exe`

## Recommended next enhancement
- Add a custom BA (Bootstrapper Application) for richer branded progress messages and support links.
