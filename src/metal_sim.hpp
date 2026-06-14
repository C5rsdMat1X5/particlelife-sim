#pragma once
#include <stdint.h>

struct Particle {
    float pos_x, pos_y;
    float vel_x, vel_y;
    float mass;
    float type;
};

struct Camera {
    float ox = 0.0f, oy = 0.0f, zoom = 1.0f;
};

struct MetalSim;
struct GLFWwindow;

MetalSim *metal_sim_create(Particle *initial, uint32_t count,
                           uint32_t num_types, uint32_t grid_dim, int srad);
void metal_render_init(MetalSim *sim, GLFWwindow *glfwWin, int fbw, int fbh);
void metal_step_and_render(MetalSim *sim, float dt, float G, float softening2,
                           float drag, Camera cam);
float *metal_sim_matrix(MetalSim *sim);
float *metal_sim_min_radius(MetalSim *sim);
float *metal_sim_max_radius(MetalSim *sim);
float *metal_sim_colors(MetalSim *sim);
void metal_sim_destroy(MetalSim *sim);
