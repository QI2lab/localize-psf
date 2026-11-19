# GPUfit installs these runtime artifacts. FindCUDAToolkit's UNKNOWN imported
# targets cannot be used by install(IMPORTED_RUNTIME_ARTIFACTS) on Linux.
include_guard(GLOBAL)
set(_localize_cuda_target "$ENV{CUDAToolkit_ROOT}/targets/x86_64-linux")
get_filename_component(_localize_cuda_state "$ENV{CUDAToolkit_ROOT}" DIRECTORY)
set(_localize_cuda_runtime "${_localize_cuda_state}/runtime")
file(MAKE_DIRECTORY "${_localize_cuda_runtime}")
foreach(_localize_cuda_library IN ITEMS cublasLt cublas)
  if(NOT TARGET CUDA::${_localize_cuda_library})
    set(_localize_cuda_location "${_localize_cuda_target}/lib/lib${_localize_cuda_library}.so.12")
    if(NOT EXISTS "${_localize_cuda_location}")
      message(FATAL_ERROR "Missing local CUDA runtime: ${_localize_cuda_location}")
    endif()
    file(REAL_PATH "${_localize_cuda_location}" _localize_cuda_location)
    # Install one regular file under the SONAME, rather than both a versioned
    # binary and its SONAME copy. Both cuBLAS libraries are large.
    set(_localize_cuda_staged "${_localize_cuda_runtime}/lib${_localize_cuda_library}.so.12")
    file(REMOVE "${_localize_cuda_staged}")
    file(CREATE_LINK "${_localize_cuda_location}" "${_localize_cuda_staged}"
      COPY_ON_ERROR RESULT _localize_cuda_link_result)
    if(NOT _localize_cuda_link_result STREQUAL "0")
      message(FATAL_ERROR "Cannot stage local CUDA runtime: ${_localize_cuda_link_result}")
    endif()
    add_library(CUDA::${_localize_cuda_library} SHARED IMPORTED GLOBAL)
    set_target_properties(CUDA::${_localize_cuda_library} PROPERTIES
      IMPORTED_LOCATION "${_localize_cuda_staged}"
      IMPORTED_SONAME "lib${_localize_cuda_library}.so.12"
      INTERFACE_INCLUDE_DIRECTORIES "${_localize_cuda_target}/include")
  endif()
endforeach()
set_property(TARGET CUDA::cublas APPEND PROPERTY INTERFACE_LINK_LIBRARIES CUDA::cublasLt)
unset(_localize_cuda_library)
unset(_localize_cuda_location)
unset(_localize_cuda_link_result)
unset(_localize_cuda_staged)
unset(_localize_cuda_runtime)
unset(_localize_cuda_state)
unset(_localize_cuda_target)
