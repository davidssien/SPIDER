#include <SDL.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <random>
#include <iomanip>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

const int WIDTH = 800;
const int HEIGHT = 800;
const int FLOOR = 750;
const float PI = 3.14159265f;

struct Vec2D {
    float x, y;

    Vec2D(float x = 0, float y = 0) : x(x), y(y) {}
    Vec2D operator+(Vec2D v) const { return {x + v.x, y + v.y}; }
    Vec2D operator-(Vec2D v) const { return {x - v.x, y - v.y}; }
    Vec2D operator*(float n) const { return {x * n, y * n}; }
    Vec2D &operator+=(Vec2D v) {
        x += v.x;
        y += v.y;
        return *this;
    }
    Vec2D &operator-=(Vec2D v) {
        x -= v.x;
        y -= v.y;
        return *this;
    }
    Vec2D &operator*=(float n) {
        x *= n;
        y *= n;
        return *this;
    }
};

Vec2D operator*(float n, Vec2D v) {
    return v * n;
}
float dot(Vec2D a, Vec2D b) {
    return a.x * b.x + a.y * b.y;
}
float cross(Vec2D a, Vec2D b) {
    return a.x * b.y - a.y * b.x;
}
Vec2D cross(float speed, Vec2D v) {
    return {-speed * v.y, speed * v.x};
}
Vec2D rotate(Vec2D v, float angle) {
    float c = std::cos(angle), s = std::sin(angle);
    return {c * v.x - s * v.y, s * v.x + c * v.y};
}

struct Bone {
    Vec2D pos, vel;
    float angle, angVel;
    float halfLength, radius, mass, inertia, invMass, invInertia;

    Bone(Vec2D p, float a, float l, float r, float m)
        : pos(p), vel(0), angle(a), angVel(0), halfLength(l), radius(r), mass(m) {
        float L = l * 2, W = r * 2;
        inertia = (m * (L * L + W * W)) / 12.f;
        invMass = 1.f / m;
        invInertia = 1.f / inertia;
    }

    Vec2D worldPoint(Vec2D local) const {
        float c = cos(angle), s = sin(angle);
        return pos + Vec2D(c * local.x - s * local.y, s * local.x + c * local.y);
    }
};

struct Joint {
    Bone *A, *B;
    Vec2D anchorA_local, anchorB_local;
    float maxTorque = 1e5f;

    float targetAngle = 0;
    float stiffness = 1.0f;

    Joint(Bone *a, Bone *b, Vec2D anchorA, Vec2D anchorB, float targetAngle = 0.0f,
          float stiffness = 1.0f)
        : A(a), B(b), anchorA_local(anchorA), anchorB_local(anchorB), targetAngle(targetAngle),
          stiffness(stiffness) {}

    void solve(float dt) {

        Vec2D rA = rotate(anchorA_local, A->angle), rB = rotate(anchorB_local, B->angle);
        Vec2D pA = A->pos + rA, pB = B->pos + rB;

        Vec2D C = pB - pA;

        Vec2D vA = A->vel + cross(A->angVel, rA);
        Vec2D vB = B->vel + cross(B->angVel, rB);

        Vec2D relVel = vB - vA;

        float beta = 0.8f;
        Vec2D bias = (beta / dt) * C;
        Vec2D Cdot = relVel + bias;

        float k11 =
            A->invMass + B->invMass + A->invInertia * rA.y * rA.y + B->invInertia * rB.y * rB.y;
        float k12 = -A->invInertia * rA.x * rA.y - B->invInertia * rB.x * rB.y;
        float k21 = k12;
        float k22 =
            A->invMass + B->invMass + A->invInertia * rA.x * rA.x + B->invInertia * rB.x * rB.x;

        float det = k11 * k22 - k12 * k21;
        if (det == 0) return;

        float invDet = 1.0f / det;

        Vec2D impulse;
        impulse.x = -(k22 * Cdot.x - k12 * Cdot.y) * invDet;
        impulse.y = -(-k21 * Cdot.x + k11 * Cdot.y) * invDet;

        A->vel -= impulse * A->invMass;
        B->vel += impulse * B->invMass;

        A->angVel -= cross(rA, impulse) * A->invInertia;
        B->angVel += cross(rB, impulse) * B->invInertia;
    }

    void applyTorque(float dt) {

        float angle = B->angle - A->angle;
        float angVel = B->angVel - A->angVel;
        float error = targetAngle - angle;
        error -= 2.0f * PI * floorf((error + PI) / (2.0f * PI));

        float k = stiffness * maxTorque;
        float d = 2.0f * sqrt(k * (A->inertia + B->inertia));

        float torque = k * error - d * angVel;

        torque = std::clamp(torque, -maxTorque, maxTorque);

        float impulse = torque * dt;

        A->angVel -= impulse * A->invInertia;
        B->angVel += impulse * B->invInertia;
    }
};

struct Skel {
    Bone bones[2] = {
        {{400, 710}, -PI / 2, 40, 12, 2},
        {{400, 630}, -PI / 2, 40, 12, 2}
    };
    Joint joints[1] = {{&bones[0], &bones[1], {40, 0}, {-40, 0}}};
    int steps = 0;
    float action = 0;

    void reset(float tilt = 0) {
        bones[0].angle = -PI / 2 + tilt;
        bones[1].angle = -PI / 2 + tilt;
        bones[0].pos = Vec2D{400, FLOOR - 12} - rotate({-28, 0}, bones[0].angle);
        bones[1].pos = bones[0].worldPoint({40, 0}) - rotate({-40, 0}, bones[1].angle);
        for (Bone& bone : bones) {
            bone.vel = {0, 0};
            bone.angVel = 0;
        }
        joints[0].targetAngle = 0;
        steps = 0;
        action = 0;
    }

    float upright() const {
        return (-std::sin(bones[0].angle) - std::sin(bones[1].angle)) / 2;
    }

    bool fallen() const {
        return upright() < 0.65f || std::abs(bones[0].pos.x - 400) > 240;
    }

    void act(float value) {
        action = std::clamp(value, -1.0f, 1.0f);
        joints[0].targetAngle = action * 0.8f;
        for (int i = 0; i < 4; i++) update(1.0f / 240.0f);
        steps++;
    }

    void writeState() const {
        for (const Bone& bone : bones) {
            std::cout << std::sin(bone.angle) << ' ' << std::cos(bone.angle) << ' '
                      << bone.angVel / 10 << ' ' << bone.vel.x / 200 << ' '
                      << bone.vel.y / 200 << ' ';
        }
        std::cout << (bones[0].pos.x - 400) / 200 << ' '
                  << (FLOOR - bones[1].pos.y) / 160 << ' ' << action << ' ';
    }

    void collide(Bone &bone, Vec2D offset, Vec2D normal, float boundary) {
        Vec2D contact = bone.pos + offset;
        float penetration = boundary + bone.radius - dot(contact, normal);
        if (penetration <= 0) return;

        Vec2D velocity = bone.vel + cross(bone.angVel, offset);
        float speed = dot(velocity, normal);
        if (speed < 0) {
            float lever = cross(offset, normal);
            float amount = -1.2f * speed / (bone.invMass + lever * lever * bone.invInertia);
            Vec2D impulse = normal * amount;
            bone.vel += impulse * bone.invMass;
            bone.angVel += cross(offset, impulse) * bone.invInertia;
        }
        bone.pos += normal * (0.8f * std::max(penetration - 0.01f, 0.0f));
    }

    void update(float dt) {
        for (Bone &bone : bones) {
            bone.vel.y += 980.6f * dt;
            bone.vel *= 1.0f / (1.0f + 0.7f * dt);
            bone.angVel *= 1.0f / (1.0f + 0.7f * dt);
        }
        for (Joint &joint : joints)
            joint.applyTorque(dt);
        for (int i = 0; i < 20; i++) {
            for (Joint &joint : joints)
                joint.solve(dt);
        }
        for (Bone &bone : bones) {
            bone.vel.x = std::clamp(bone.vel.x, -6000.0f, 6000.0f);
            bone.vel.y = std::clamp(bone.vel.y, -6000.0f, 6000.0f);
            bone.angVel = std::clamp(bone.angVel, -50.0f, 50.0f);
            bone.pos += bone.vel * dt;
            bone.angle += bone.angVel * dt;

            for (float end : {-1.0f, 1.0f}) {
                Vec2D offset = rotate({end * (bone.halfLength - bone.radius), 0}, bone.angle);
                collide(bone, offset, {0, -1}, -FLOOR);
                collide(bone, offset, {0, 1}, 0);
                collide(bone, offset, {1, 0}, 0);
                collide(bone, offset, {-1, 0}, -WIDTH);
            }
        }
    }
};

struct Engine {
    SDL_Window *window = nullptr;
    SDL_Renderer *renderer = nullptr;
    Skel bots[9];
    Uint32 seen[9] = {};
    SDL_Texture* ghostLayer = nullptr;
    bool running = true;
    bool paused = false;
    int connection = -1;
    int selectedModel = 0;
    int displayedModel = -1;
    sockaddr_in server{};
    Uint32 lastPacket = 0;
    std::string command = "watch";

    bool init() {
        if (SDL_Init(SDL_INIT_VIDEO) != 0) return false;

        window = SDL_CreateWindow("Balance | Waiting for Python server | 1-9: model | Space: pause | R: reset | P: push",
                                  SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, 0);
        if (!window) return false;

        renderer = SDL_CreateRenderer(window, -1, 0);
        if (!renderer) return false;

        for (Skel& bot : bots) bot.reset();
        ghostLayer = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888,
                                       SDL_TEXTUREACCESS_TARGET, WIDTH, HEIGHT);
        if (!ghostLayer) return false;
        SDL_SetTextureBlendMode(ghostLayer, SDL_BLENDMODE_BLEND);
        SDL_SetTextureAlphaMod(ghostLayer, 65);
        connection = socket(AF_INET, SOCK_DGRAM, 0);
        if (connection < 0) return false;
        server.sin_family = AF_INET;
        server.sin_port = htons(5005);
        inet_pton(AF_INET, "127.0.0.1", &server.sin_addr);
        if (connect(connection, reinterpret_cast<sockaddr*>(&server), sizeof(server)) != 0)
            return false;
        return true;
    }

    void handleEvents() {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) running = false;
            if (event.type == SDL_KEYDOWN && !event.key.repeat) {
                if (event.key.keysym.sym >= SDLK_1 && event.key.keysym.sym <= SDLK_9)
                    selectedModel = event.key.keysym.sym - SDLK_1;
                switch (event.key.keysym.sym) {
                case SDLK_ESCAPE:
                    running = false;
                    break;
                case SDLK_SPACE:
                    paused = !paused;
                    break;
                case SDLK_r:
                    command = "reset";
                    break;
                case SDLK_p:
                    command = "push";
                    break;
                }
            }
        }
    }

    void receive() {
        std::string request = std::to_string(selectedModel) + " " + command;
        send(connection, request.data(), request.size(), 0);
        command = "watch";
        char packet[1024];
        int size;
        while ((size = recv(connection, packet, sizeof(packet) - 1, MSG_DONTWAIT)) > 0) {
            packet[size] = 0;
            std::istringstream input(packet);
            std::string version, name;
            int model, episode;
            long trained;
            float reward;
            float x0, y0, a0, x1, y1, a1;
            if (!(input >> version >> model >> name >> trained >> episode >> reward
                        >> x0 >> y0 >> a0 >> x1 >> y1 >> a1) || version != "SPIDER1") continue;
            if (model < 0 || model >= 9) continue;
            if (!std::isfinite(x0 + y0 + a0 + x1 + y1 + a1)) continue;
            if (!paused) {
                bots[model].bones[0].pos = {x0, y0}; bots[model].bones[0].angle = a0;
                bots[model].bones[1].pos = {x1, y1}; bots[model].bones[1].angle = a1;
            }
            seen[model] = SDL_GetTicks();
            if (model != selectedModel) continue;
            if (displayedModel != model) {
                std::cout << "Watching " << name << " (" << trained << " training steps)" << std::endl;
                displayedModel = model;
            }
            lastPacket = SDL_GetTicks();
            std::string title = name + " | trained steps: " + std::to_string(trained)
                + " | episode step: " + std::to_string(episode)
                + " | 1-9: model | Space: pause view | R: reset | P: push";
            SDL_SetWindowTitle(window, title.c_str());
        }
        if (SDL_GetTicks() - lastPacket > 2000)
            SDL_SetWindowTitle(window, "Waiting for Python: python3 train.py | Esc: quit");
    }

    void drawCircle(Vec2D center, int radius) {
        int x = std::lround(center.x), y = std::lround(center.y);
        for (int row = -radius; row <= radius; row++) {
            int halfWidth = std::sqrt(radius * radius - row * row);
            SDL_RenderDrawLine(renderer, x - halfWidth, y + row, x + halfWidth, y + row);
        }
    }

    void drawBone(const Bone& bone) {
        Vec2D start = bone.worldPoint({-bone.halfLength + bone.radius, 0});
        Vec2D end = bone.worldPoint({bone.halfLength - bone.radius, 0});
        Vec2D difference = end - start;
        int steps = std::max(1, int(std::ceil(std::sqrt(dot(difference, difference)))));
        for (int i = 0; i <= steps; i++)
            drawCircle(start + difference * (float(i) / steps), bone.radius);
    }

    void drawBot(const Skel& bot) {
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
        for (const Bone& bone : bot.bones) drawBone(bone);
        SDL_SetRenderDrawColor(renderer, 145, 145, 145, 255);
        drawCircle(bot.joints[0].A->worldPoint(bot.joints[0].anchorA_local), 4);
    }

    void draw() {
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
        SDL_RenderDrawLine(renderer, 0, FLOOR, WIDTH, FLOOR);

        for (int model = 0; model < 9; model++) {
            if (model == selectedModel || !seen[model]) continue;
            if (SDL_GetTicks() - seen[model] > 2000) continue;
            SDL_SetRenderTarget(renderer, ghostLayer);
            SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0);
            SDL_RenderClear(renderer);
            drawBot(bots[model]);
            SDL_SetRenderTarget(renderer, nullptr);
            SDL_RenderCopy(renderer, ghostLayer, nullptr, nullptr);
        }
        drawBot(bots[selectedModel]);
        SDL_RenderPresent(renderer);
    }

    int run() {
        if (!init()) {
            std::cerr << "Viewer initialization failed: " << SDL_GetError() << '\n';
            return 1;
        }
        while (running) {
            Uint32 frameStart = SDL_GetTicks();
            handleEvents();
            if (!running) break;
            receive();
            draw();
            Uint32 elapsed = SDL_GetTicks() - frameStart;
            if (elapsed < 16) SDL_Delay(16 - elapsed);
        }
        return 0;
    }

    ~Engine() {
        if (connection >= 0) close(connection);
        SDL_DestroyTexture(ghostLayer);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
    }
};

int physics(int count, unsigned seed) {
    if (count < 1 || count > 256) return 1;
    std::mt19937 random(seed);
    std::uniform_real_distribution<float> tilt(-0.08f, 0.08f);
    std::vector<std::unique_ptr<Skel>> worlds;
    for (int i = 0; i < count; i++) {
        worlds.push_back(std::make_unique<Skel>());
        worlds.back()->reset(tilt(random));
    }
    std::cout << std::setprecision(8);
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream input(line);
        std::string command;
        input >> command;
        if (command == "quit") break;
        if (command != "step" && command != "reset" && command != "push") return 1;
        int selected = -1;
        if (command != "step") input >> selected;
        int index = 0;
        for (auto& world : worlds) {
            bool chosen = selected < 0 || selected == index;
            index++;
            float reward = 0;
            bool done = false;
            if (command == "reset" && chosen) world->reset(tilt(random));
            if (command == "push" && chosen) world->bones[1].vel.x += 35;
            if (command == "step") {
                float action;
                if (!(input >> action) || !std::isfinite(action)) return 1;
                world->act(action);
                done = world->fallen() || world->steps >= 600;
                reward = world->fallen() ? -5 : 1 + world->upright() - 0.02f * action * action;
            }
            int episodeSteps = world->steps;
            float pose[6] = {world->bones[0].pos.x, world->bones[0].pos.y, world->bones[0].angle,
                             world->bones[1].pos.x, world->bones[1].pos.y, world->bones[1].angle};
            if (done) world->reset(tilt(random));
            world->writeState();
            std::cout << reward << ' ' << done << ' ' << episodeSteps << ' ';
            for (float value : pose) std::cout << value << ' ';
            std::cout << '\n';
        }
        std::cout << std::flush;
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc == 4 && std::string(argv[1]) == "--physics") {
        try { return physics(std::stoi(argv[2]), std::stoul(argv[3])); }
        catch (...) { return 1; }
    }
    Engine engine;
    return engine.run();
}
