cmake_minimum_required(VERSION 3.26)
project(ops_sgx LANGUAGES C CXX)

# ————————————————————————————————————————————#
# 1) Build settings
# ————————————————————————————————————————————#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Make sure SGX_SDK is set (via `source /usr/sgxsdk/environment`)
if(NOT DEFINED ENV{SGX_SDK})
  message(FATAL_ERROR "Please source /usr/sgxsdk/environment first")
endif()
set(SGX_SDK $ENV{SGX_SDK})

# Find the SGX edger8r & signer tools
find_program(SGX_EDGER8R sgx_edger8r HINTS ${SGX_SDK}/bin)
find_program(SGX_SIGN    sgx_sign    HINTS ${SGX_SDK}/bin)
if(NOT SGX_EDGER8R)
  message(FATAL_ERROR "Could not find sgx_edger8r")
endif()
if(NOT SGX_SIGN)
  message(FATAL_ERROR "Could not find sgx_sign")
endif()

# ————————————————————————————————————————————#
# 2) Generate SGX stubs from your EDL
# ————————————————————————————————————————————#
set(EDL_FILE ${CMAKE_SOURCE_DIR}/Enclave/Enclave.edl)
set(EDL_OUT  ${CMAKE_BINARY_DIR}/edl)
file(MAKE_DIRECTORY ${EDL_OUT})

add_custom_command(
  OUTPUT
    ${EDL_OUT}/Enclave_t.c
    ${EDL_OUT}/Enclave_t.h
    ${EDL_OUT}/Enclave_u.c
    ${EDL_OUT}/Enclave_u.h
  COMMAND ${SGX_EDGER8R}
          --trusted   ${EDL_FILE}
          --search-path ${CMAKE_SOURCE_DIR}/Enclave
          --search-path ${SGX_SDK}/include
  COMMAND ${SGX_EDGER8R}
          --untrusted ${EDL_FILE}
          --search-path ${CMAKE_SOURCE_DIR}/Enclave
          --search-path ${SGX_SDK}/include
  WORKING_DIRECTORY ${EDL_OUT}
  DEPENDS ${EDL_FILE}
  COMMENT "Generating SGX EDL stubs"
)

include_directories(
  ${EDL_OUT}
  ${SGX_SDK}/include
)

# ————————————————————————————————————————————#
# 3) Build the enclave shared object
# ————————————————————————————————————————————#
add_library(OpsEnclave SHARED
  ${EDL_OUT}/Enclave_t.c
  Enclave/Enclave.cpp
)

# Link in the SGX trusted runtime libraries
target_link_libraries(OpsEnclave PRIVATE
  ${SGX_SDK}/lib64/libsgx_trts.a
  ${SGX_SDK}/lib64/libsgx_tcrypto.a
  ${SGX_SDK}/lib64/libsgx_tservice.a
)

# 4) Sign the enclave after build
add_custom_command(TARGET OpsEnclave POST_BUILD
  COMMAND ${SGX_SIGN}
          sign
          -key     ${CMAKE_SOURCE_DIR}/Enclave/Enclave_private.pem
          -enclave $<TARGET_FILE:OpsEnclave>
          -out     $<TARGET_FILE_DIR:OpsEnclave>/Enclave.signed.so
  COMMENT "Signing SGX enclave"
)

# ————————————————————————————————————————————#
# 5) Build the host‐side PyTorch extension
# ————————————————————————————————————————————#
find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

add_library(ops_sgx SHARED
  ${EDL_OUT}/Enclave_u.c
  App/ops_sim.cpp
  App/ops_sgx.cpp
)

target_include_directories(ops_sgx PRIVATE
  ${Torch_INCLUDE_DIRS}
  ${Python3_INCLUDE_DIRS}
  ${SGX_SDK}/include
  ${EDL_OUT}
)

# Plain‐signature link order: signed enclave stub, SGX URTS, PyTorch, Python
target_link_libraries(ops_sgx
  OpsEnclave
  sgx_urts
  ${TORCH_LIBRARIES}
  ${Python3_LIBRARIES}
)

target_compile_definitions(ops_sgx PRIVATE
  ${TORCH_DEFINITIONS}
)

# Ensure enclave is built & signed before the extension links against it
add_dependencies(ops_sgx OpsEnclave)