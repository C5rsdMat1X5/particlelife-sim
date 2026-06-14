#include "metal_sim.hpp"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>


static const uint32_t GRID_DIM = 256;
static const int SEARCH_RAD = 8;

static const uint32_t N = 1 << 16;
static const uint32_t NUM_TYPES = 16;
static const float G = 5e-7f;
static const float SOFTENING2 = 1e-5f;
static const float DRAG = 0.95f;
static const int WINDOW_W = 900;
static const int WINDOW_H = 900;
static const float RAND_SCALE = 4.0f;


static const float MAX_SEARCH = (float)SEARCH_RAD * (2.0f / (float)GRID_DIM);

static void hsv_to_rgb(float h, float s, float v, float &r, float &g,
                       float &b) {
    int i = (int)(h * 6.0f) % 6;
    float f = h * 6.0f - (int)(h * 6.0f);
    float p = v * (1.0f - s);
    float q = v * (1.0f - f * s);
    float t = v * (1.0f - (1.0f - f) * s);
    switch (i) {
    case 0:
        r = v;
        g = t;
        b = p;
        break;
    case 1:
        r = q;
        g = v;
        b = p;
        break;
    case 2:
        r = p;
        g = v;
        b = t;
        break;
    case 3:
        r = p;
        g = q;
        b = v;
        break;
    case 4:
        r = t;
        g = p;
        b = v;
        break;
    default:
        r = v;
        g = p;
        b = q;
        break;
    }
}

struct InputState {
    Camera cam;
    bool panning = false;
    double last_x = 0.0, last_y = 0.0;
    int fb_w = WINDOW_W, fb_h = WINDOW_H;
    float dt_scale = 1.0f;
};

static void screen_to_ndc(const InputState &s, double px, double py, float &nx,
                          float &ny) {
    nx = (float)(px / s.fb_w * 2.0 - 1.0);
    ny = -(float)(py / s.fb_h * 2.0 - 1.0);
}

static void scroll_cb(GLFWwindow *win, double, double dy) {
    auto *s = (InputState *)glfwGetWindowUserPointer(win);
    double cx, cy;
    glfwGetCursorPos(win, &cx, &cy);
    float nx, ny;
    screen_to_ndc(*s, cx, cy, nx, ny);
    float wx = nx / s->cam.zoom + s->cam.ox;
    float wy = ny / s->cam.zoom + s->cam.oy;
    s->cam.zoom *= (dy > 0) ? 1.12f : (1.0f / 1.12f);
    s->cam.zoom = std::clamp(s->cam.zoom, 0.05f, 200.0f);
    s->cam.ox = wx - nx / s->cam.zoom;
    s->cam.oy = wy - ny / s->cam.zoom;
}

static void mouse_btn_cb(GLFWwindow *win, int btn, int action, int) {
    auto *s = (InputState *)glfwGetWindowUserPointer(win);
    if (btn == GLFW_MOUSE_BUTTON_LEFT) {
        s->panning = (action == GLFW_PRESS);
        glfwGetCursorPos(win, &s->last_x, &s->last_y);
    }
}

static void cursor_cb(GLFWwindow *win, double x, double y) {
    auto *s = (InputState *)glfwGetWindowUserPointer(win);
    if (!s->panning)
        return;
    float dx = (float)(x - s->last_x) / s->fb_w * 2.0f / s->cam.zoom;
    float dy = -(float)(y - s->last_y) / s->fb_h * 2.0f / s->cam.zoom;
    s->cam.ox -= dx;
    s->cam.oy -= dy;
    s->last_x = x;
    s->last_y = y;
}

static void print_matrix(const char *label, const float *mat, uint32_t n) {
    printf("\n%s (%u×%u)\n", label, n, n);
    printf("      ");
    for (uint32_t j = 0; j < n; j++)
        printf(" T%-6u", j);
    printf("\n");
    for (uint32_t i = 0; i < n; i++) {
        printf("  T%u  ", i);
        for (uint32_t j = 0; j < n; j++)
            printf(" %+.4f", mat[i * n + j]);
        printf("\n");
    }
}

int main() {
    unsigned seed = (unsigned)std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count();
    srand(seed);
    printf("Seed: %u\n", seed);

    std::vector<Particle> particles(N);
    for (uint32_t i = 0; i < N; i++) {
        float angle = (float)rand() / (float)RAND_MAX * 2.0f * (float)M_PI;
        float r = sqrtf((float)rand() / (float)RAND_MAX) * 0.85f;
        particles[i] = {r * cosf(angle),
                        r * sinf(angle),
                        0.0f,
                        0.0f,
                        1.0f,
                        (float)(rand() % NUM_TYPES)};
    }


    MetalSim *sim =
        metal_sim_create(particles.data(), N, NUM_TYPES, GRID_DIM, SEARCH_RAD);
    if (!sim)
        return 1;

    float *m = metal_sim_matrix(sim);
    float *mnR = metal_sim_min_radius(sim);
    float *mxR = metal_sim_max_radius(sim);
    float *c = metal_sim_colors(sim);

    for (uint32_t i = 0; i < NUM_TYPES * NUM_TYPES; i++)
        m[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * RAND_SCALE;

    for (uint32_t i = 0; i < NUM_TYPES * NUM_TYPES; i++) {
        float lo = ((float)rand() / (float)RAND_MAX) * MAX_SEARCH * 0.4f;
        float hi = lo + ((float)rand() / (float)RAND_MAX) * (MAX_SEARCH - lo);
        mnR[i] = lo;
        mxR[i] = hi;
    }

    for (uint32_t t = 0; t < NUM_TYPES; t++) {
        float hue = (float)t / (float)NUM_TYPES;
        hsv_to_rgb(hue, 0.85f, 1.0f, c[t * 4 + 0], c[t * 4 + 1], c[t * 4 + 2]);
        c[t * 4 + 3] = 1.0f;
    }

    print_matrix("Interaction matrix", m, NUM_TYPES);
    print_matrix("Min radius", mnR, NUM_TYPES);
    print_matrix("Max radius", mxR, NUM_TYPES);
    printf("\n");

    if (!glfwInit()) {
        fprintf(stderr, "glfwInit failed\n");
        return 1;
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    GLFWwindow *win =
        glfwCreateWindow(WINDOW_W, WINDOW_H, "N-Body", nullptr, nullptr);
    if (!win) {
        fprintf(stderr, "glfwCreateWindow failed\n");
        return 1;
    }

    InputState input;
    glfwGetFramebufferSize(win, &input.fb_w, &input.fb_h);
    glfwSetWindowUserPointer(win, &input);
    glfwSetScrollCallback(win, scroll_cb);
    glfwSetMouseButtonCallback(win, mouse_btn_cb);
    glfwSetCursorPosCallback(win, cursor_cb);

    metal_render_init(sim, win, input.fb_w, input.fb_h);

    double prev = glfwGetTime();
    int frame = 0;

    while (!glfwWindowShouldClose(win)) {
        double now = glfwGetTime();
        double raw_dt = now - prev;
        prev = now;

        if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(win, GLFW_TRUE);
        if (glfwGetKey(win, GLFW_KEY_R) == GLFW_PRESS)
            input.cam = Camera{};
        if (glfwGetKey(win, GLFW_KEY_0) == GLFW_PRESS)
            input.dt_scale = 1.0f;
        if (glfwGetKey(win, GLFW_KEY_UP) == GLFW_PRESS)
            input.dt_scale = std::min(
                input.dt_scale * (float)std::pow(2.0, raw_dt * 2.0), 32.0f);
        if (glfwGetKey(win, GLFW_KEY_DOWN) == GLFW_PRESS)
            input.dt_scale = std::max(
                input.dt_scale / (float)std::pow(2.0, raw_dt * 2.0), 0.0f);

        double dt_real = raw_dt * (double)input.dt_scale;
        metal_step_and_render(sim, (float)dt_real, G, SOFTENING2, DRAG,
                              input.cam);
        glfwPollEvents();

        if (++frame % 2 == 0) {
            char title[128];
            snprintf(title, sizeof(title),
                     "N-Body  %uK particles  %u types  %.0f FPS  %ux%u grid  "
                     "%.2f dt  %.2f zoom",
                     N / 1000, NUM_TYPES, raw_dt > 0 ? 1.0 / raw_dt : 0.0,
                     GRID_DIM, GRID_DIM, input.dt_scale, (double)input.cam.zoom);
            glfwSetWindowTitle(win, title);
        }
    }

    metal_sim_destroy(sim);
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
