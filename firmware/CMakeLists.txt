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
