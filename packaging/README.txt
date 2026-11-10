Vathography App - depth maps from photographs, without ComfyUI.

Start: double-click "Vathography App.bat", or drop photos or a folder on it.

The first start needs two downloads, each started by your click in the window. The models (about 15.5 GB) go into the models folder next to the app; if you already have them (for example in a ComfyUI models folder), choose "I already have them" instead. PyTorch and OpenCV (about 2.9 GB) come from their publishers, pytorch.org and PyPI, and are installed in the third_party folder next to the app; they are not included in this download and are used under their own licences (see NOTICE.txt).

Depth Estimation tab: drop photos or a folder, choose Base or See-through Depth Estimation and the output format, then Start. See-through is the default: it sometimes gives better results than Base, for example detail in the background that Base loses, so trying both on a photo can be worthwhile. Each depth map is saved next to its photo (or in a folder you choose) as soon as it is done.

Relief Accentuation tab: drop a depth map or an export from the Vathography Blender add-on, choose a strength from 1 to 6, then Start.

Needs: Windows 10 or 11 (64-bit), an NVIDIA graphics card of the RTX 30-series or newer with NVIDIA driver 570.65 or newer, and about 25 GB of free disk space during the first start. The app's own folder needs about 8.3 GB free while PyTorch and OpenCV are installed, even when the models are on another drive. The app keeps its graphics memory under 6.5 GB and starts with as little as about 5 GB free, which is meant to fit 6 GB cards; it has passed on a real 6 GB card, not yet on an 8 GB one. With less graphics memory the depth map can differ very slightly: measured on one photo with the Base model, by at most 0.01 on its 0 to 1 scale; on a photo the sky repair works on, it can differ more. System memory in use peaks at about 21 GB for a 33 MP photo and about 24 GB for 66 MP; a PC with less needs a large enough Windows page file.

Sky repair: the depth model can read a fine web of cables over a sky as a ceiling. On photos large enough to be split into tiles, the app checks for this and repairs what it finds, automatically. A repaired photo takes up to about four times longer.

Known limits of the sky repair: it can miss a part that went wrong, some cables can stay missing or faint, or come out slightly thin, and it can add small spots on parts that were fine.

Depth maps and reliefs are numbered _v001, _v002 and so on; a new run never overwrites an earlier file.

The depth maps are estimated from the photograph; they are relative depth, never measurements.

To update the app: unzip the new version into a new, empty folder (not over the old one), then move the models folder and the third_party folder from the old version into it (or point "I already have them" at the models). If the new version uses another PyTorch or OpenCV version, it downloads that one again.

To report a problem: in the app, open the gear menu, then Troubleshooting log, copy the log and paste it in a new issue at https://github.com/vathography/Vathography-App/issues. Issues are public and the log shows your folder and file names, so look through it first.

Licence: Vathography App is free software under the GNU Affero General Public License version 3 only (AGPL-3.0-only), with additional permissions (see LICENSE). Its complete source code, including the scripts that build this download, is at https://github.com/vathography/Vathography-App, tag app-0.1.7. The works it includes and the software it downloads keep their own licences (see NOTICE.txt and the licences folder).

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. This applies to the extent permitted by applicable law.

https://vathography.com
