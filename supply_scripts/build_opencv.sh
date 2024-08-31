VERSION=4.8.0

# test -e ${VERSION}.zip || wget https://github.com/opencv/opencv/archive/refs/tags/${VERSION}.zip
# test -e opencv-${VERSION} || unzip ${VERSION}.zip

# test -e opencv_extra_${VERSION}.zip || wget -O opencv_extra_${VERSION}.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/${VERSION}.zip
# test -e opencv_contrib-${VERSION} || unzip opencv_extra_${VERSION}.zip


cd opencv-${VERSION}
mkdir build
cd build

# change it before start, use "locate" ibcudnn.so.9.3.0 or you can install without cuda supply 
# -D CUDNN_LIBRARY=/usr/lib/x86_64-linux-gnu/libcudnn.so.9.3.0 \
# -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.3 \

# cmake -D CMAKE_BUILD_TYPE=RELEASE \
# -D CMAKE_INSTALL_PREFIX=/usr/local \
# -D WITH_TBB=ON \
# -D WITH_V4L=ON \
# -D WITH_GTK=ON \
# -D GTK_INCLUDE_DIRS=/usr/include/gtk-2.0 \
# -D GTK_LIBRARIES=/usr/lib/x86_64-linux-gnu/libgtk-x11-2.0.so \
# -D BUILD_opencv_cudacodec=ON \
# -D WITH_QT=ON \
# -D WITH_OPENGL=ON \
# -D BUILD_opencv_apps=OFF \
# -D BUILD_opencv_python2=OFF \
# -D OPENCV_GENERATE_PKGCONFIG=ON \
# -D OPENCV_PC_FILE_NAME=opencv.pc \
# -D OPENCV_ENABLE_NONFREE=ON \
# -D INSTALL_PYTHON_EXAMPLES=OFF \
# -D INSTALL_C_EXAMPLES=OFF \
# -D BUILD_EXAMPLES=OFF \
# -D WITH_FFMPEG=ON \
# ..

cmake -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_OPENGL=ON \
   -D WITH_GTK=ON -D GTK_INCLUDE_DIRS=/usr/include/gtk-2.0 \
   -D GTK_LIBRARIES=/usr/lib/x86_64-linux-gnu/libgtk-x11-2.0.so \
   -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..

sudo make -j 12
sudo make -j 12 install
