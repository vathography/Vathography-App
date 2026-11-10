Source of the FFmpeg libraries in this download

The FFmpeg libraries in runtime\Lib\site-packages\av.libs (avcodec, avdevice, avfilter, avformat, avutil, swresample, swscale) are FFmpeg 8.1.1, licensed under the GNU Lesser General Public License version 2.1 or later (COPYING.LGPLv2.1 in this folder). They are separate DLL files.

ffmpeg-n8.1.1.tar.gz in this folder is their complete source: FFmpeg's git tag n8.1.1 (commit 239f2c733de417201d7ad3b3b8b0d9b63285b2b1), as published at https://github.com/FFmpeg/FFmpeg. sha256 9ecb4cc05c1bdf996513b162aca48d5d934e000eae29e07763d69f5f4c77fa70.

They were built unmodified, with MSYS2 MINGW64 (gcc 16.2.0, binutils 2.47, MinGW-w64 runtime 14.0.0, nasm 3.02, zlib 1.3.2 linked statically), with this configuration:

./configure --arch=x86_64 --target-os=mingw32 --enable-shared --disable-static --disable-programs --disable-doc --disable-debug --disable-autodetect --disable-everything --disable-network --enable-zlib --enable-x86asm --enable-decoder=mjpeg,png,tiff,webp,exr --enable-encoder=png,exr --enable-demuxer=image2,image2pipe,'image_*_pipe' --enable-parser=mjpeg,png,webp --enable-protocol=file,pipe --enable-filter=buffer,buffersink,pad,fillborders,format,scale,null --extra-ldflags='-static-libgcc -static'

then make and make install. The Python binding (PyAV 17.1.0, BSD-3-Clause, https://github.com/PyAV-Org/PyAV) was compiled against them from its published source.
