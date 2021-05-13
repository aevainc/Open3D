#!/usr/bin/env bash

set -eu
/home/yixing/.dotfiles/bin/clean_build
NPROC=$(nproc)
echo NPROC="$NPROC"
set +u
DEVELOPER_BUILD="ON"
set -u
if [[ "$DEVELOPER_BUILD" != "OFF" ]]; then # Validate input coming from GHA input field
    DEVELOPER_BUILD=ON
    DOC_ARGS=""
else
    DOC_ARGS="--is_release"
    echo "Building docs for a new Open3D release"
    echo
    echo "Building Open3D with ENABLE_HEADLESS_RENDERING=ON for Jupyter notebooks"
    echo
fi
cmakeOptions=("-DDEVELOPER_BUILD=$DEVELOPER_BUILD"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DBUILD_JUPYTER_EXTENSION=OFF"
    "-DWITH_OPENMP=ON"
    "-DOPEN3D_THIRD_PARTY_DOWNLOAD_DIR=$HOME/open3d_downloads"
    "-DCMAKE_INSTALL_PREFIX=~/open3d_install"
)
set -x # Echo commands on
cmake "${cmakeOptions[@]}" \
    -DENABLE_HEADLESS_RENDERING=ON \
    -DBUILD_GUI=OFF \
    ..
pip uninstall -y open3d
make -j$NPROC || make -j$NPROC || make -j$NPROC || make -j$NPROC
make install-pip-package -j$NPROC
bin/GLInfo
python -c "from open3d import *; import open3d; print(open3d)"
cd ../docs # To Open3D/docs
python make_docs.py $DOC_ARGS --clean_notebooks --execute_notebooks=always --pyapi_rst=never
python -m pip uninstall --yes open3d
cd ../build
set +x # Echo commands off
echo
echo "Building Open3D with BUILD_GUI=ON for visualization.{gui,rendering} documentation"
echo
set -x # Echo commands on
cmake "${cmakeOptions[@]}" \
    -DENABLE_HEADLESS_RENDERING=OFF \
    -DBUILD_GUI=ON \
    ..
make -j$NPROC || make -j$NPROC || make -j$NPROC || make -j$NPROC
make install-pip-package -j$NPROC
bin/GLInfo || echo "Expect failure since HEADLESS_RENDERING=OFF"
python -c "from open3d import *; import open3d; print(open3d)"
cd ../docs # To Open3D/docs
python make_docs.py $DOC_ARGS --pyapi_rst=always --execute_notebooks=never --sphinx --doxygen
set +x # Echo commands off
