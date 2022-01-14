include(ExternalProject)

ExternalProject_Add(
    ext_sciplot
    PREFIX sciplot
#     URL https://github.com/sciplot/sciplot/archive/refs/heads/master.zip
#     URL_HASH SHA256=e0612b28dfac6b4d3e8fb308aedab9ecd13878f8e9c004a54f17ea741130d42f
    URL https://github.com/sciplot/sciplot/archive/refs/tags/v0.1.0.zip
    URL_HASH SHA256=133cc63a0f0674173f41c914fda8b37d33b40207d89dfe94aa35bfa444288815
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/sciplot"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_sciplot SOURCE_DIR)
set(SCIPLOT_INCLUDE_DIRS ${SOURCE_DIR}/)
