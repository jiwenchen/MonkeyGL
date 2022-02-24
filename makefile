CXX = g++
LDFLAGS = -fPIC -shared
INC = `python3 -m pybind11 --includes`
INC_CUDA = -I./cuda_common
EXT_SUFFIX = `python3-config --extension-suffix`
NVCCFLAGS = /usr/local/cuda/bin/nvcc -Xcompiler -fPIC -ccbin g++ $(INC_CUDA) $(INC) -O3 -shared -std=c++11 -Xcompiler -fPIC -m64

ifeq ($(dbg),1)
	NVCCFLAGS += -g -G
	TARGET := debug
else
	TARGET := release
endif

SRC = ./core/PybindInterface.cpp ./core/DeviceInfo.cpp ./core/Defines.cpp ./core/Point.cpp ./core/Direction.cpp
OBJ =
EXEC =

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CXX)

cpp:
	mkdir -p $(TARGET)
	$(NVCCFLAGS) $(SRC) -o ./$(TARGET)/MonkeyGL$(EXT_SUFFIX)

clean:
	rm -rf ./$(TARGET)/*.so ./$(TARGET)/*.o