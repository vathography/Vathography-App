Vathography App for macOS - depth maps from photographs, without ComfyUI.

By Karim Joseph Nassar, developed with Claude Code.

Start: double-click "Vathography App", or drop photos or a folder on its icon.

The app is not notarized by Apple, so the first time you open it macOS will say it cannot verify it: go to System Settings > Privacy & Security, click "Open Anyway" once, and after that it opens normally.

The first start needs two downloads, each started by your click in the window. The models (about 15.5 GB) go into the folder Library/Application Support/Vathography App/models in your home folder; if you already have them (for example in a ComfyUI models folder), choose "I already have them" instead. PyTorch and OpenCV (about 0.13 GB to download, about 0.46 GB installed) come from PyPI, the Python package index where their publishers release them, and are installed in Library/Application Support/Vathography App/third_party; they are not included in this download and are used under their own licences (see NOTICE.txt).

Depth Estimation tab: drop photos or a folder, choose Base or See-through Depth Estimation and the output format, then Start. See-through is the default: it sometimes gives better results than Base, for example detail in the background that Base loses, so trying both on a photo can be worthwhile. Each depth map is saved next to its photo (or in a folder you choose) as soon as it is done.

Relief Accentuation tab: drop a depth map or an export from the Vathography Blender add-on, choose a strength from 1 to 6, then Start.

Needs: a Mac with Apple silicon (M1 or later) and macOS 14 Sonoma or newer; Macs with an Intel processor are not supported. Memory: tested on a Mac with 24 GB, up to a 3456 x 3456 photo; Macs with less memory are not tested. Time per photo on that Mac (M4 Pro): about 4 minutes for a 1000-pixel photo, about 1 hour for a 3456 x 3456 photo. The free disk space during the first start is not yet measured on a Mac.

Sky repair: the depth model can read a fine web of cables over a sky as a ceiling. On photos large enough to be split into tiles, the app checks for this and repairs what it finds, automatically. A repaired photo takes up to about four times longer.

Known limits of the sky repair: it can miss a part that went wrong, some cables can stay missing or faint, or come out slightly thin, and it can add small spots on parts that were fine.

Depth maps and reliefs are numbered _v001, _v002 and so on; a new run never overwrites an earlier file.

The depth maps are estimated from the photograph; they are relative depth, never measurements.

To update the app: replace the old app with the new one. The models, PyTorch and OpenCV stay in Library/Application Support/Vathography App, where the new version finds them; if it uses another PyTorch or OpenCV version, it downloads that one again.

To report a problem: in the app, open the gear menu, then Troubleshooting log, copy the log and paste it in a new issue at https://github.com/vathography/Vathography-App/issues. Issues are public and the log shows your folder and file names, so look through it first.

Licence: Vathography App is free software under the GNU Affero General Public License version 3 only (AGPL-3.0-only), with additional permissions (see LICENSE). Its complete source code, including the scripts that build this download, is at https://github.com/vathography/Vathography-App, tag app-0.2.0. The works it includes and the software it downloads keep their own licences (see NOTICE.txt and the licences folder).

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. This applies to the extent permitted by applicable law.

https://vathography.com
