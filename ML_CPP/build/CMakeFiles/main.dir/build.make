# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.31

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\CMake\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\CMake\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\44225\Desktop\ML_CPP

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\44225\Desktop\ML_CPP\build

# Include any dependencies generated for this target.
include CMakeFiles/main.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/main.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/main.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/main.dir/flags.make

CMakeFiles/main.dir/codegen:
.PHONY : CMakeFiles/main.dir/codegen

CMakeFiles/main.dir/main.cpp.obj: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/main.cpp.obj: CMakeFiles/main.dir/includes_CXX.rsp
CMakeFiles/main.dir/main.cpp.obj: C:/Users/44225/Desktop/ML_CPP/main.cpp
CMakeFiles/main.dir/main.cpp.obj: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=C:\Users\44225\Desktop\ML_CPP\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/main.dir/main.cpp.obj"
	D:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/main.cpp.obj -MF CMakeFiles\main.dir\main.cpp.obj.d -o CMakeFiles\main.dir\main.cpp.obj -c C:\Users\44225\Desktop\ML_CPP\main.cpp

CMakeFiles/main.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/main.dir/main.cpp.i"
	D:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\44225\Desktop\ML_CPP\main.cpp > CMakeFiles\main.dir\main.cpp.i

CMakeFiles/main.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/main.dir/main.cpp.s"
	D:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\44225\Desktop\ML_CPP\main.cpp -o CMakeFiles\main.dir\main.cpp.s

CMakeFiles/main.dir/common/metrics.cpp.obj: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/common/metrics.cpp.obj: CMakeFiles/main.dir/includes_CXX.rsp
CMakeFiles/main.dir/common/metrics.cpp.obj: C:/Users/44225/Desktop/ML_CPP/common/metrics.cpp
CMakeFiles/main.dir/common/metrics.cpp.obj: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=C:\Users\44225\Desktop\ML_CPP\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/main.dir/common/metrics.cpp.obj"
	D:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/common/metrics.cpp.obj -MF CMakeFiles\main.dir\common\metrics.cpp.obj.d -o CMakeFiles\main.dir\common\metrics.cpp.obj -c C:\Users\44225\Desktop\ML_CPP\common\metrics.cpp

CMakeFiles/main.dir/common/metrics.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/main.dir/common/metrics.cpp.i"
	D:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\44225\Desktop\ML_CPP\common\metrics.cpp > CMakeFiles\main.dir\common\metrics.cpp.i

CMakeFiles/main.dir/common/metrics.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/main.dir/common/metrics.cpp.s"
	D:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\44225\Desktop\ML_CPP\common\metrics.cpp -o CMakeFiles\main.dir\common\metrics.cpp.s

CMakeFiles/main.dir/common/utils.cpp.obj: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/common/utils.cpp.obj: CMakeFiles/main.dir/includes_CXX.rsp
CMakeFiles/main.dir/common/utils.cpp.obj: C:/Users/44225/Desktop/ML_CPP/common/utils.cpp
CMakeFiles/main.dir/common/utils.cpp.obj: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=C:\Users\44225\Desktop\ML_CPP\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/main.dir/common/utils.cpp.obj"
	D:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/common/utils.cpp.obj -MF CMakeFiles\main.dir\common\utils.cpp.obj.d -o CMakeFiles\main.dir\common\utils.cpp.obj -c C:\Users\44225\Desktop\ML_CPP\common\utils.cpp

CMakeFiles/main.dir/common/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/main.dir/common/utils.cpp.i"
	D:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\44225\Desktop\ML_CPP\common\utils.cpp > CMakeFiles\main.dir\common\utils.cpp.i

CMakeFiles/main.dir/common/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/main.dir/common/utils.cpp.s"
	D:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\44225\Desktop\ML_CPP\common\utils.cpp -o CMakeFiles\main.dir\common\utils.cpp.s

CMakeFiles/main.dir/linear_regression/linearRegression.cpp.obj: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/linear_regression/linearRegression.cpp.obj: CMakeFiles/main.dir/includes_CXX.rsp
CMakeFiles/main.dir/linear_regression/linearRegression.cpp.obj: C:/Users/44225/Desktop/ML_CPP/linear_regression/linearRegression.cpp
CMakeFiles/main.dir/linear_regression/linearRegression.cpp.obj: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=C:\Users\44225\Desktop\ML_CPP\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/main.dir/linear_regression/linearRegression.cpp.obj"
	D:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/linear_regression/linearRegression.cpp.obj -MF CMakeFiles\main.dir\linear_regression\linearRegression.cpp.obj.d -o CMakeFiles\main.dir\linear_regression\linearRegression.cpp.obj -c C:\Users\44225\Desktop\ML_CPP\linear_regression\linearRegression.cpp

CMakeFiles/main.dir/linear_regression/linearRegression.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/main.dir/linear_regression/linearRegression.cpp.i"
	D:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\44225\Desktop\ML_CPP\linear_regression\linearRegression.cpp > CMakeFiles\main.dir\linear_regression\linearRegression.cpp.i

CMakeFiles/main.dir/linear_regression/linearRegression.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/main.dir/linear_regression/linearRegression.cpp.s"
	D:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\44225\Desktop\ML_CPP\linear_regression\linearRegression.cpp -o CMakeFiles\main.dir\linear_regression\linearRegression.cpp.s

CMakeFiles/main.dir/svm/svm.cpp.obj: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/svm/svm.cpp.obj: CMakeFiles/main.dir/includes_CXX.rsp
CMakeFiles/main.dir/svm/svm.cpp.obj: C:/Users/44225/Desktop/ML_CPP/svm/svm.cpp
CMakeFiles/main.dir/svm/svm.cpp.obj: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=C:\Users\44225\Desktop\ML_CPP\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/main.dir/svm/svm.cpp.obj"
	D:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/svm/svm.cpp.obj -MF CMakeFiles\main.dir\svm\svm.cpp.obj.d -o CMakeFiles\main.dir\svm\svm.cpp.obj -c C:\Users\44225\Desktop\ML_CPP\svm\svm.cpp

CMakeFiles/main.dir/svm/svm.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/main.dir/svm/svm.cpp.i"
	D:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\44225\Desktop\ML_CPP\svm\svm.cpp > CMakeFiles\main.dir\svm\svm.cpp.i

CMakeFiles/main.dir/svm/svm.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/main.dir/svm/svm.cpp.s"
	D:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\44225\Desktop\ML_CPP\svm\svm.cpp -o CMakeFiles\main.dir\svm\svm.cpp.s

# Object files for target main
main_OBJECTS = \
"CMakeFiles/main.dir/main.cpp.obj" \
"CMakeFiles/main.dir/common/metrics.cpp.obj" \
"CMakeFiles/main.dir/common/utils.cpp.obj" \
"CMakeFiles/main.dir/linear_regression/linearRegression.cpp.obj" \
"CMakeFiles/main.dir/svm/svm.cpp.obj"

# External object files for target main
main_EXTERNAL_OBJECTS =

main.exe: CMakeFiles/main.dir/main.cpp.obj
main.exe: CMakeFiles/main.dir/common/metrics.cpp.obj
main.exe: CMakeFiles/main.dir/common/utils.cpp.obj
main.exe: CMakeFiles/main.dir/linear_regression/linearRegression.cpp.obj
main.exe: CMakeFiles/main.dir/svm/svm.cpp.obj
main.exe: CMakeFiles/main.dir/build.make
main.exe: D:/Users/44225/anaconda3/libs/python311.lib
main.exe: CMakeFiles/main.dir/linkLibs.rsp
main.exe: CMakeFiles/main.dir/objects1.rsp
main.exe: CMakeFiles/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=C:\Users\44225\Desktop\ML_CPP\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable main.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\main.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/main.dir/build: main.exe
.PHONY : CMakeFiles/main.dir/build

CMakeFiles/main.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\main.dir\cmake_clean.cmake
.PHONY : CMakeFiles/main.dir/clean

CMakeFiles/main.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\44225\Desktop\ML_CPP C:\Users\44225\Desktop\ML_CPP C:\Users\44225\Desktop\ML_CPP\build C:\Users\44225\Desktop\ML_CPP\build C:\Users\44225\Desktop\ML_CPP\build\CMakeFiles\main.dir\DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/main.dir/depend

