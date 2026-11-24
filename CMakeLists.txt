cmake_minimum_required(VERSION 3.24)
project(PA_AHNTS VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(WIN32)
    add_compile_definitions(WIN32_LEAN_AND_MEAN NOMINMAX)
endif()

if(MSVC)
    add_compile_options(/W4 /utf-8)
else()
    add_compile_options(-Wall -Wextra)
endif()

add_subdirectory(tools/env_probe)

add_executable(pa_ahnts_bench
    src/core/dx12_context.cpp
    src/core/scene_geometry.cpp
    src/telemetry/telemetry_logger.cpp
    src/baselines/material_pack.cpp
    src/baselines/baseline_runner.cpp
    src/ram_tier/ram_tiered_pool.cpp
    src/baselines/main.cpp
)

target_link_libraries(pa_ahnts_bench PRIVATE
    d3d12
    dxgi
    dxguid
    uuid
    ole32
    gdi32
    user32
    pdh
)

if(NOT MSVC)
    target_link_options(pa_ahnts_bench PRIVATE -Wl,--export-all-symbols)
endif()

set(DXC_EXE "${CMAKE_SOURCE_DIR}/third_party/dxc/build/native/bin/x64/dxc.exe")
set(AGILITY_BIN_DIR "${CMAKE_SOURCE_DIR}/third_party/agility_sdk/build/native/bin/x64")
set(SHADER_SRC_DIR "${CMAKE_SOURCE_DIR}/src/baselines/shaders")
set(SHADER_OUT_DIR "${CMAKE_BINARY_DIR}/shaders")
set(AGILITY_OUT_DIR "$<TARGET_FILE_DIR:pa_ahnts_bench>/D3D12")

add_custom_command(TARGET pa_ahnts_bench POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${AGILITY_OUT_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${AGILITY_BIN_DIR}/D3D12Core.dll" "${AGILITY_OUT_DIR}/D3D12Core.dll"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${AGILITY_BIN_DIR}/d3d12SDKLayers.dll" "${AGILITY_OUT_DIR}/d3d12SDKLayers.dll"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${SHADER_OUT_DIR}"
    COMMAND "${DXC_EXE}" -T vs_6_6 -E main -HV 2021 -O3 -Fo "${SHADER_OUT_DIR}/pbr_vs.cso" "${SHADER_SRC_DIR}/pbr_vs.hlsl"
    COMMAND "${DXC_EXE}" -T ps_6_6 -E main -HV 2021 -O3 -Fo "${SHADER_OUT_DIR}/pbr_bcn.ps.cso" "${SHADER_SRC_DIR}/pbr_bcn.ps.hlsl"
    COMMAND "${DXC_EXE}" -T ps_6_6 -E main -HV 2021 -O3 -enable-16bit-types -Fo "${SHADER_OUT_DIR}/pbr_ntc_sample.ps.cso" "${SHADER_SRC_DIR}/pbr_ntc_sample.ps.hlsl"
    COMMAND "${DXC_EXE}" -T cs_6_6 -E main -HV 2021 -O3 -enable-16bit-types -Fo "${SHADER_OUT_DIR}/ntc_decode_bcn.cs.cso" "${SHADER_SRC_DIR}/ntc_decode_bcn.cs.hlsl"
    COMMENT "Copying Agility SDK DLLs and compiling SM 6.6 HLSL shaders"
)
