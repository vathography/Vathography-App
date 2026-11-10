# Vathography App

High-resolution depth maps estimated from high-resolution photographs, in a standalone app. An easy-to-use alternative to the Vathography ComfyUI nodes, for artists who do not use ComfyUI. The app is free. 
Windows app for NVIDIA graphics cards (RTX 30-series or newer).

This repository holds the complete source code of Vathography App, including the scripts that build the download. Each release has a tag: `app-0.1.7` is the source of version 0.1.7.

## What is here

- `vathography_app/` - the app.
- `packaging/build_package.py` - builds the portable download (Python runtime, packages, app, launcher).
- `packaging/build_av_lgpl.py` - builds the PyAV package the download uses, on a minimal FFmpeg under the LGPL (no GPL code).
- `packaging/asset_originals/` - the original of the image the app ships in processed form, and the script that processes it.
- `packaging/README.txt`, `packaging/NOTICE.txt`, `packaging/licences/` - the texts that ship inside the download.

## Problems

Report a problem in [Issues](https://github.com/vathography/Vathography-App/issues). The app's Troubleshooting log (gear menu) helps; it shows your folder and file names, so look through it first.

## Licence

Vathography App is free software under the GNU Affero General Public License version 3 only (AGPL-3.0-only), with additional permissions (see LICENSE). The works it includes and the software it downloads keep their own licences (see `packaging/NOTICE.txt` and `packaging/licences/`).

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. This applies to the extent permitted by applicable law.

https://vathography.com
