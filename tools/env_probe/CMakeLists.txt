add_executable(dxgi_dml_probe dxgi_dml_probe.cpp)

target_link_libraries(dxgi_dml_probe PRIVATE
    d3d12
    dxgi
    dxguid
    uuid
    ole32
)
