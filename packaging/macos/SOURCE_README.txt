Source of the LGPL libraries in this download

FFmpeg

The FFmpeg libraries in Vathography App.app/Contents/Resources/runtime/lib/python3.10/site-packages/av/.dylibs (avcodec, avdevice, avfilter, avformat, avutil, swresample, swscale) are FFmpeg 8.1.1, licensed under the GNU Lesser General Public License version 2.1 or later (COPYING.LGPLv2.1 in this folder). They are separate .dylib files.

ffmpeg-n8.1.1.tar.gz in this folder is their complete source: FFmpeg's git tag n8.1.1 (commit 239f2c733de417201d7ad3b3b8b0d9b63285b2b1), as published at https://github.com/FFmpeg/FFmpeg. sha256 9ecb4cc05c1bdf996513b162aca48d5d934e000eae29e07763d69f5f4c77fa70.

They were built unmodified on macOS with {CLANG_VERSION}, for macOS 14.0 or newer, linked to the zlib of macOS (a system library, not included in this download), with this configuration:

./configure --arch=arm64 --target-os=darwin --cc=clang --enable-shared --disable-static --disable-programs --disable-doc --disable-debug --disable-autodetect --disable-everything --disable-network --enable-zlib --enable-decoder=mjpeg,png,tiff,webp,exr --enable-encoder=png,exr --enable-demuxer=image2,image2pipe,'image_*_pipe' --enable-parser=mjpeg,png,webp --enable-protocol=file,pipe --enable-filter=buffer,buffersink,pad,fillborders,format,scale,null --extra-cflags=-mmacosx-version-min=14.0 --extra-ldflags=-mmacosx-version-min=14.0

then make and make install. The Python binding (PyAV 17.1.0, BSD-3-Clause, https://github.com/PyAV-Org/PyAV) was compiled against them from its published source, and the libraries were copied into it with delocate 0.13.0.

libquadmath

libquadmath.0.dylib in Vathography App.app/Contents/Resources/runtime/lib/python3.10/site-packages/scipy/.dylibs, included in SciPy 1.15.3's macOS wheel as published on PyPI, is the GCC Quad-Precision Math Library, licensed under the GNU Lesser General Public License version 2.1 or later (COPYING.LGPLv2.1 in this folder). It is a separate .dylib file.

The build paths recorded in the library show that it was built from GCC 13.3.0 for aarch64-apple-darwin23, as in Homebrew's gcc@13 13.3.0 package of 2024-05-22, which applies Homebrew's patch for Apple silicon. Its source is the libquadmath folder of that patched tree:
- gcc-13.3.0.tar.xz in this folder: GCC 13.3.0 as published at https://ftp.gnu.org/gnu/gcc/gcc-13.3.0/, sha256 0845e9621c9543a13f484e94584a49ffc0129970e9914624235fc1d061a0c083.
- gcc-13.3.0.diff in this folder: Homebrew's patch, as published at https://raw.githubusercontent.com/Homebrew/formula-patches/bda0faddfbfb392e7b9c9101056b2c5ab2500508/gcc/gcc-13.3.0.diff, sha256 c5e9236430ef6edbdda7de9ac70bf79e21628077a48322cec7f3f064ccfc243d.
