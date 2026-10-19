# Build the C++ engine. Header-only engine + one translation unit, so the
# same sources build with a single compiler command anywhere (see README.md).
#
#   make          optimized binary: bullpen_cpp (bullpen_cpp.exe on Windows)
#   make debug    ASan + UBSan binary: bullpen_cpp_debug
#   make test     build, then run the selftest
#   make clean

CXX      ?= g++
STD      := -std=c++17
WARN     := -Wall -Wextra -Wpedantic -Werror
RELEASE  := -O3 -DNDEBUG
DEBUG    := -O0 -g -fsanitize=address,undefined -fno-omit-frame-pointer

ifeq ($(OS),Windows_NT)
EXE := .exe
endif

BIN      := bullpen_cpp$(EXE)
DBG      := bullpen_cpp_debug$(EXE)
SRCS     := main.cpp
HDRS     := orderbook.hpp

.PHONY: all debug test clean

all: $(BIN)

$(BIN): $(SRCS) $(HDRS)
	$(CXX) $(STD) $(RELEASE) $(WARN) $(CXXFLAGS) -o $@ $(SRCS) $(LDFLAGS)

debug: $(DBG)

$(DBG): $(SRCS) $(HDRS)
	$(CXX) $(STD) $(DEBUG) $(WARN) $(CXXFLAGS) -o $@ $(SRCS) $(LDFLAGS)

test: $(BIN)
	./$(BIN) selftest

clean:
	rm -f $(BIN) $(DBG)
