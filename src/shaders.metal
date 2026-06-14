#include <metal_stdlib>
using namespace metal;

#ifndef GDIM
#define GDIM 256
#endif

#ifndef SRAD
#define SRAD 8
#endif

#ifndef MAX_TYPES
#define MAX_TYPES 8
#endif

struct Particle {
    float2 pos;
    float2 vel;
    float mass;
    float type;
};

struct GridParams {
    uint N;
    uint num_types;
    float inv_cell;
    float dt;
    float G;
    float softening2;
    float drag;
};

inline int to_cell(float p, float inv_cell) {
    return clamp((int)((p + 1.0f) * inv_cell), 0, (int)GDIM - 1);
}

kernel void clear_cells(device atomic_uint *counts [[buffer(0)]],
                        uint id [[thread_position_in_grid]]) {
    atomic_store_explicit(&counts[id], 0, memory_order_relaxed);
}

kernel void count_cells(device const Particle *src [[buffer(0)]],
                        device atomic_uint *counts [[buffer(1)]],
                        constant GridParams &gp [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= gp.N)
        return;
    int cx = to_cell(src[id].pos.x, gp.inv_cell);
    int cy = to_cell(src[id].pos.y, gp.inv_cell);
    atomic_fetch_add_explicit(&counts[cy * GDIM + cx], 1, memory_order_relaxed);
}

kernel void scatter_particles(device const Particle *src [[buffer(0)]],
                              device Particle *sorted [[buffer(1)]],
                              device atomic_uint *cell_write [[buffer(2)]],
                              constant GridParams &gp [[buffer(3)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= gp.N)
        return;
    int cx = to_cell(src[id].pos.x, gp.inv_cell);
    int cy = to_cell(src[id].pos.y, gp.inv_cell);
    uint slot = atomic_fetch_add_explicit(&cell_write[cy * GDIM + cx], 1,
                                          memory_order_relaxed);
    sorted[slot] = src[id];
}

kernel void grid_force(device const Particle *sorted [[buffer(0)]],
                       device Particle *output [[buffer(1)]],
                       constant uint *starts [[buffer(2)]],
                       constant uint *counts [[buffer(3)]],
                       constant GridParams &gp [[buffer(4)]],
                       constant float *matrix [[buffer(5)]],
                       constant float *min_rad [[buffer(6)]],
                       constant float *max_rad [[buffer(7)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= gp.N)
        return;


    const Particle self = sorted[id];
    const float2 pos = self.pos;
    const int myType = (int)self.type;




    float local_g_matrix[MAX_TYPES];
    float local_min_rad_sq[MAX_TYPES];
    float local_max_rad_sq[MAX_TYPES];

    const uint type_limit = min(gp.num_types, (uint)MAX_TYPES);
    for (uint t = 0; t < type_limit; t++) {
        uint pair = myType * gp.num_types + t;
        local_g_matrix[t] = matrix[pair] * gp.G;

        float r_min = min_rad[pair];
        float r_max = max_rad[pair];
        local_min_rad_sq[t] = r_min * r_min;
        local_max_rad_sq[t] = r_max * r_max;
    }

    const int cx = to_cell(pos.x, gp.inv_cell);
    const int cy = to_cell(pos.y, gp.inv_cell);

    float2 accel = float2(0.0f);


    for (int dy = -SRAD; dy <= SRAD; dy++) {
        int ny = cy + dy;
        if (ny < 0 || ny >= (int)GDIM)
            continue;
        for (int dx = -SRAD; dx <= SRAD; dx++) {
            int nx = cx + dx;
            if (nx < 0 || nx >= (int)GDIM)
                continue;

            uint cell = ny * GDIM + nx;
            uint start = starts[cell];
            uint cnt = counts[cell];

            for (uint k = 0; k < cnt; k++) {

                const Particle other = sorted[start + k];
                const int nType = (int)other.type;

                if (nType >= (int)type_limit)
                    continue;

                float2 d = other.pos - pos;
                float dist2 = dot(d, d);


                if (dist2 < local_min_rad_sq[nType] || dist2 > local_max_rad_sq[nType])
                    continue;


                float r2 = dist2 + gp.softening2;
                float inv_r = rsqrt(r2);
                float inv3 = inv_r * inv_r * inv_r;


                accel += d * (other.mass * inv3 * local_g_matrix[nType]);
            }
        }
    }


    output[id].vel = (self.vel + accel * gp.dt) * gp.drag;
    output[id].pos = self.pos + output[id].vel * gp.dt;
    output[id].mass = self.mass;
    output[id].type = self.type;
}

struct Camera {
    float ox, oy, zoom;
};

struct VertOut {
    float4 position [[position]];
    float point_size [[point_size]];
    float3 color;
};

vertex VertOut particle_vert(device const Particle *particles [[buffer(0)]],
                             constant Camera &cam [[buffer(1)]],
                             device const float4 *colors [[buffer(2)]],
                             uint vid [[vertex_id]]) {
    float2 world = particles[vid].pos;
    float2 view = (world - float2(cam.ox, cam.oy)) * cam.zoom;
    VertOut o;
    o.position = float4(view, 0.0, 1.0);
    o.point_size = clamp(2.0 * cam.zoom, 1.0, 16.0);
    o.color = colors[(int)particles[vid].type].rgb;
    return o;
}

fragment float4 particle_frag(VertOut in [[stage_in]],
                              float2 coord [[point_coord]]) {
    float2 c = coord - 0.5;
    float d = dot(c, c) * 4.0;
    if (d > 1.0)
        discard_fragment();
    float alpha = 0.8 * (1.0 - d);
    return float4(in.color * alpha, alpha);
}
