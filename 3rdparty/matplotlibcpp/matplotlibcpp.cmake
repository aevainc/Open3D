include(ExternalProject)

ExternalProject_Add(
    ext_matplotlibcpp
    PREFIX matplotlibcpp
#     URL https://github.com/lava/matplotlib-cpp/archive/refs/heads/master.zip
#     URL_HASH SHA256=591fa8e50bca3994e9f408fb8b930716acaebf86c3cc8814125127cfbc7800ae
    URL https://github.com/Cryoris/matplotlib-cpp/archive/refs/heads/master.zip
    URL_HASH SHA256=7b72e552380db1650aae6edd1f808976b5be4e845f41804790c4a29527db20b9
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/matplotlibcpp"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_matplotlibcpp SOURCE_DIR)
set(MATPLOTLIBCPP_INCLUDE_DIRS ${SOURCE_DIR}/)
