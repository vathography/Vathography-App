<img width="1280" height="640" alt="Vathography App" src="https://github.com/user-attachments/assets/be8f1877-a04c-4f4d-a5bb-776afd850262" />

# Vathography App

High-resolution depth maps estimated from high-resolution photographs, in a standalone app. An easy-to-use alternative to the Vathography ComfyUI nodes, for artists who do not use ComfyUI. The app is free. 
Windows app for NVIDIA graphics cards (RTX 30-series or newer); Mac app for Apple silicon (M1 or later, macOS 14 or newer), tested on a Mac with 24 GB of memory; smaller Macs are untested.

By Karim Joseph Nassar, developed with Claude Code.

This repository holds the complete source code of Vathography App, including the scripts that build the download. Each release has a tag: `app-0.1.8` is the source of the Windows version 0.1.8, `app-0.2.0` the source of the Mac version 0.2.0.

## What is here

- `vathography_app/` - the app.
- `packaging/build_package.py` - builds the portable download (Python runtime, packages, app, launcher).
- `packaging/build_av_lgpl.py` - builds the PyAV package the download uses, on a minimal FFmpeg under the LGPL (no GPL code).
- `packaging/asset_originals/` - the original of the image the app ships in processed form, and the script that processes it.
- `packaging/README.txt`, `packaging/NOTICE.txt`, `packaging/licences/` - the texts that ship inside the download.
- `packaging/build_package_mac.py`, `packaging/build_av_lgpl_mac.py`, `packaging/macos/`, `packaging/licences_macos/` - the same for the Mac download; `.github/workflows/build-mac.yml` builds it on GitHub's macOS runners and `packaging/smoke_mac.py` checks the built zip there.
- `packaging/rust_crates/`, `packaging/bundled_licences/` - the scripts that collect licence texts for `licences/` and `licences_macos/`; `packaging/hf_models/` - the card and notice of the Hugging Face repository the app downloads its models from; `packaging/make_icon.py` - makes the app icon.

## Problems

Report a problem in [Issues](https://github.com/vathography/Vathography-App/issues). The app's Troubleshooting log (gear menu) helps; it shows your folder and file names, so look through it first.

## Licence

Vathography App is free software under the GNU Affero General Public License version 3 only (AGPL-3.0-only), with additional permissions (see LICENSE). The works it includes and the software it downloads keep their own licences (see `packaging/NOTICE.txt` and `packaging/licences/`).

The software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software. This applies to the extent permitted by applicable law.

https://vathography.com
