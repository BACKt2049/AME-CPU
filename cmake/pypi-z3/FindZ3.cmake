if(Z3_FOUND)
    return()
endif()

set(_tilelang_z3_roots)
foreach(_root
    "${Z3_ROOT}"
    "$ENV{Z3_ROOT}"
    "$ENV{Z3_DIR}"
    "$ENV{Z3_HOME}"
)
    if(_root)
        list(APPEND _tilelang_z3_roots "${_root}")
    endif()
endforeach()

find_path(
    Z3_INCLUDE_DIR
    NAMES z3++.h
    HINTS ${_tilelang_z3_roots}
    PATH_SUFFIXES include
)
find_library(
    Z3_LIBRARY
    NAMES z3 libz3
    HINTS ${_tilelang_z3_roots}
    PATH_SUFFIXES bin lib lib64
)

if(Z3_INCLUDE_DIR AND Z3_LIBRARY)
    message(STATUS "Found Z3 from explicit/system paths")
else()
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import z3; print(z3.__path__[0])"
        OUTPUT_VARIABLE Z3_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE Z3_PYTHON_RESULT
    )
    if(NOT Z3_PYTHON_RESULT EQUAL 0 OR Z3_PATH STREQUAL "")
        message(FATAL_ERROR
            "Failed to locate Z3. Set Z3_ROOT to a Z3 install prefix or install the z3 Python package "
            "(for example z3-solver>=4.13.0 on supported platforms).")
    endif()
    message(STATUS "Find Z3 in Python package path: ${Z3_PATH}")
    find_path(Z3_INCLUDE_DIR NO_DEFAULT_PATH NAMES z3++.h PATHS ${Z3_PATH}/include)
    find_library(Z3_LIBRARY NO_DEFAULT_PATH NAMES z3 libz3 PATHS ${Z3_PATH}/bin ${Z3_PATH}/lib ${Z3_PATH}/lib64)
endif()

message(STATUS "Found Z3 include dir: ${Z3_INCLUDE_DIR}")
message(STATUS "Found Z3 library: ${Z3_LIBRARY}")

if(NOT Z3_INCLUDE_DIR OR NOT Z3_LIBRARY)
    message(FATAL_ERROR "Could not find Z3 library or include directory")
endif()

if(NOT TARGET z3::libz3)
    add_library(z3::libz3 SHARED IMPORTED GLOBAL)
    set_target_properties(z3::libz3
        PROPERTIES
        IMPORTED_LOCATION ${Z3_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES ${Z3_INCLUDE_DIR}
    )
endif()

set(Z3_CXX_INCLUDE_DIRS ${Z3_INCLUDE_DIR})
set(Z3_C_INCLUDE_DIRS ${Z3_INCLUDE_DIR})
set(Z3_FOUND TRUE)
