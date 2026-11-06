# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.3.1)
set(toolchainVersion 15_2_Rel1)
set(picotoolVersion 2.3.1)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================
cmake_minimum_required(VERSION 3.13)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

set(PICO_BOARD pico2 CACHE STRING "Board type")

# Finds the SDK through PICO_SDK_PATH (CI) or the VS Code Pico extension's install.
include(pico_sdk_import.cmake)

project(andon C CXX ASM)

pico_sdk_init()

set(ANDON_SOURCES
    src/main.c
    src/andon_light.c
    src/button.c
)
add_executable(andon ${ANDON_SOURCES})

# Strict warnings on our files only; the SDK sources compiled into the target are left alone.
set_source_files_properties(${ANDON_SOURCES} PROPERTIES COMPILE_OPTIONS "-Wall;-Wextra;-Werror")

target_include_directories(andon PRIVATE src)
target_link_libraries(andon pico_stdlib)

# printf goes over the USB cable as a serial port; no UART adapter needed.
pico_enable_stdio_usb(andon 1)
pico_enable_stdio_uart(andon 0)

# Writes andon.uf2 (drag onto the board in BOOTSEL mode) next to the .elf.
pico_add_extra_outputs(andon)

# andon_bench.uf2: the C log-mel and int8 CNN checked against the Python test vectors and
# timed on the board. Needs the headers from andon.export_c (real or --synthetic), which are
# generated, not in git; without them only the andon target is built.
set(ANDON_MODEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/model CACHE PATH "andon.export_c output")
if(EXISTS ${ANDON_MODEL_DIR}/andon_model.h AND EXISTS ${ANDON_MODEL_DIR}/andon_test_vectors.h)
    set(BENCH_SOURCES
        src/bench.c
        src/selftest.c
        src/logmel.c
        src/model.c
        src/model_params.c
    )
    add_executable(andon_bench ${BENCH_SOURCES})
    # The generated headers define every array static; each file uses only some of them.
    set_source_files_properties(${BENCH_SOURCES} PROPERTIES
        COMPILE_OPTIONS "-Wall;-Wextra;-Werror;-Wno-unused-const-variable")
    target_include_directories(andon_bench PRIVATE src ${ANDON_MODEL_DIR})
    target_link_libraries(andon_bench pico_stdlib)
    pico_enable_stdio_usb(andon_bench 1)
    pico_enable_stdio_uart(andon_bench 0)
    pico_add_extra_outputs(andon_bench)
else()
    message(STATUS "No model headers in ${ANDON_MODEL_DIR}: skipping andon_bench "
                   "(uv run python -m andon.export_c ... --out firmware/model)")
endif()
