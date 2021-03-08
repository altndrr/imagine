PROJECT_NAME = imagine
CUDA_PATH = /usr/local/cuda

CC = g++
CC_FLAGS =
NVCC = nvcc
NVCC_FLAGS =

CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
CUDA_LINK_LIBS= -lcudart

OBJ_DIR = bin
SRC_DIR = src
INC_DIR = include

OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/common.o

# Link c++ and CUDA compiled object files to target executable:
$(PROJECT_NAME) : $(OBJS)
	$(CC) $(CC_FLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

# Compile main .cpp file to object files:
$(OBJ_DIR)/%.o : %.cpp
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile C++ source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp include/%.h
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

# Removes the "bin" and "obj" folders
clean:
	rm -rf $(OBJ_DIR)/*.o
	rm ./$(PROJECT_NAME)
