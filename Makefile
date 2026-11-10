CXX = clang++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra
SDLFLAGS = $(shell sdl2-config --cflags --libs)

app: src/main.cpp Makefile
	$(CXX) $(CXXFLAGS) src/main.cpp -o app $(SDLFLAGS)

run: app
	./app

clean:
	rm -f app

.PHONY: run clean
