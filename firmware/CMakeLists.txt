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

# The generated headers from andon.export_c and andon.export_gmm (real or --synthetic); they
# come from training data, so they are not in git.
set(ANDON_MODEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/model CACHE PATH "andon.export_c/export_gmm output")
# The detector in andon.uf2 needs the front end and the per-fan GMM.
foreach(header andon_frontend.h andon_gmm.h)
    if(NOT EXISTS ${ANDON_MODEL_DIR}/${header})
        message(FATAL_ERROR "Missing ${ANDON_MODEL_DIR}/${header}: run uv run python -m "
                            "andon.export_gmm --config configs/h3_gmm.yaml --seed 0 --out "
                            "firmware/model (or --synthetic)")
    endif()
endforeach()

set(ANDON_SOURCES
    src/main.c
    src/andon_light.c
    src/button.c
    src/command.c
    src/detector.c
    src/fan.c
    src/frontend_params.c
    src/gmm.c
    src/gmm_params.c
    src/logmel.c
    src/mic.c
    src/mic_stream.c
    src/recorder.c
    src/say.c
)
add_executable(andon ${ANDON_SOURCES})
pico_generate_pio_header(andon ${CMAKE_CURRENT_SOURCE_DIR}/src/mic.pio)

# Strict warnings on our files only; the SDK sources compiled into the target are left alone.
# The generated headers define every array static; each file uses only some of them.
set_source_files_properties(${ANDON_SOURCES} PROPERTIES
    COMPILE_OPTIONS "-Wall;-Wextra;-Werror;-Wno-unused-const-variable")

# andon_version.h names the git commit in the boot line and the status reply.
set(ANDON_VERSION_H ${CMAKE_CURRENT_BINARY_DIR}/generated/andon_version.h)
add_custom_target(andon_version ALL
    COMMAND ${CMAKE_COMMAND} -DSRC=${CMAKE_CURRENT_SOURCE_DIR} -DOUT=${ANDON_VERSION_H}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/andon_version.cmake
    BYPRODUCTS ${ANDON_VERSION_H})
add_dependencies(andon andon_version)
target_include_directories(andon PRIVATE src ${CMAKE_CURRENT_BINARY_DIR}/generated ${ANDON_MODEL_DIR})
target_link_libraries(andon pico_stdlib hardware_pwm hardware_pio hardware_dma hardware_watchdog)
# A USB transmit buffer that holds several mic frames (526 B each), so the stream and the
# text lines can check for room and never wait on the PC (recorder.h, say.h). The SDK's
# default is 64 B.
target_compile_definitions(andon PRIVATE CFG_TUD_CDC_TX_BUFSIZE=4096)

# printf goes over the USB cable as a serial port; no UART adapter needed.
pico_enable_stdio_usb(andon 1)
pico_enable_stdio_uart(andon 0)

# Writes andon.uf2 (drag onto the board in BOOTSEL mode) next to the .elf.
pico_add_extra_outputs(andon)

# andon_bench.uf2: the C log-mel, the int8 CNN and the per-fan GMMs checked against the
# Python test vectors and timed on the board. Needs andon.export_c's headers as well as
# andon.export_gmm's; without them only the andon target is built.
set(ANDON_MODEL_HEADERS andon_frontend.h andon_model.h andon_test_vectors.h andon_gmm.h
    andon_gmm_vectors.h)
set(ANDON_HAVE_HEADERS TRUE)
foreach(header ${ANDON_MODEL_HEADERS})
    if(NOT EXISTS ${ANDON_MODEL_DIR}/${header})
        set(ANDON_HAVE_HEADERS FALSE)
    endif()
endforeach()
if(ANDON_HAVE_HEADERS)
    set(BENCH_SOURCES
        src/bench.c
        src/selftest.c
        src/logmel.c
        src/frontend_params.c
        src/model.c
        src/model_params.c
        src/gmm.c
        src/gmm_params.c
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
    message(STATUS "Missing model headers in ${ANDON_MODEL_DIR}: skipping andon_bench "
                   "(uv run python -m andon.export_c ... and andon.export_gmm ... "
                   "--out firmware/model)")
endif()
