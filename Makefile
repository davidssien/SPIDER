CXX = clang++
CXXFLAGS = -std=c++17 -Wall -Wextra
SDLFLAGS = $(shell sdl2-config --cflags --libs)

TARGET = app
SRC = $(wildcard src/*.cpp)

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(SDLFLAGS)

run: all
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
