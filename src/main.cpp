
#include <SDL.h>
#include <iostream>
using namespace std;

struct Vec2 {
    float x;
    float y;

    Vec2(float x, float y) : x(x), y(y) {}
};

struct Engine {
    SDL_Window* window;
    SDL_Renderer* renderer;

    int WIDTH = 800;
    int HEIGHT = 800;

    Engine() {
        SDL_Init(SDL_INIT_VIDEO);
        window = SDL_CreateWindow(
            "Render Test",
            SDL_WINDOWPOS_CENTERED,
            SDL_WINDOWPOS_CENTERED,
            WIDTH, HEIGHT, 0
        );

        renderer = SDL_CreateRenderer(window, -1, 0);
    }

    void DrawCircle(int centerX, int centerY, int radius) {
        for (int w = 0; w < radius * 2; w++) {
            for (int h = 0; h < radius * 2; h++) {
                int dx = radius - w;
                int dy = radius - h;

                if (dx * dx + dy * dy <= radius * radius) {
                    SDL_RenderDrawPoint(renderer, centerX + dx, centerY + dy);
                }
            }
        }
    }

    void run() {
        bool running = true;
        SDL_Event event;

        while (running) {
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT)
                    running = false;
            }

            // reset screen
            SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
            SDL_RenderClear(renderer);

            // Draw a white circle
            SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
            DrawCircle(250, 250, 100);

            // Display everything
            SDL_RenderPresent(renderer);
        }
    }

    ~Engine() {
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
    }
};

int main() {
    cout << "RENDER TEST" << endl;

    Engine engine;
    engine.run();

    return 0;
}
