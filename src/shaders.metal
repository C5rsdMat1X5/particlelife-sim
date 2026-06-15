#include <metal_stdlib>
using namespace metal;

struct Particle {
    float2 pos;
    float2 vel;
    float mass;
    float type;
};

#ifndef GDIM
#define GDIM 512
#endif

#ifndef SRAD
#define SRAD 8
#endif

#ifndef MAX_TYPES
#define MAX_TYPES 16
#endif

#ifndef REPEL_FORCE
#define REPEL_FORCE 1.0f
#endif

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
    atomic_fetch_add_explicit(&counts[(uint)cy * GDIM + (uint)cx], 1, memory_order_relaxed);
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
    uint slot = atomic_fetch_add_explicit(&cell_write[(uint)cy * GDIM + (uint)cx], 1,
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

    const uint base = (uint)myType * gp.num_types;

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

            uint cell = (uint)ny * GDIM + (uint)nx;
            uint start = starts[cell];
            uint end = start + counts[cell];

            for (uint k = start; k < end; k++) {
                if (k == id)
                    continue;
                int nType = (int)sorted[k].type;
                float2 d = sorted[k].pos - pos;
                float dist2 = dot(d, d);

                uint pair = base + (uint)nType;
                float mn = min_rad[pair];

                float mx = max_rad[pair];

                if (dist2 > mx * mx)
                    continue;

                float r2 = dist2 + gp.softening2;
                float inv_r = rsqrt(r2);
                float dist = r2 * inv_r;
                if (dist < mn) {
                    float force = dist / mn - 1.0f;
                    accel += d * (force * REPEL_FORCE * inv_r);
                } else {
                    float inv3 = inv_r * inv_r * inv_r;
                    accel += d * (gp.G * sorted[k].mass * matrix[pair] * inv3);
                }
            }
        }
    }

    float2 vel = fma(accel,gp.dt,self.vel) * gp.drag;
    output[id].pos = fma(vel,gp.dt,self.pos);
    output[id].vel = vel;
    output[id].mass = self.mass;
    output[id].type = self.type;
}

// ── Render ────────────────────────────────────────────────────────────────────

struct Camera {
    float ox, oy, zoom, aspect;
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
    view.x /= cam.aspect;
    VertOut o;
    o.position = float4(view, 0.0, 1.0);
    o.point_size = cam.zoom * 4.0;
    o.color = colors[(int)particles[vid].type].rgb;
    return o;
}

fragment float4 particle_frag(VertOut in [[stage_in]],
                              float2 coord [[point_coord]]) {
    float2 c = coord - 0.5f;
    float r = length(c) * 2.0f;

    if (r > 1.0f)
        discard_fragment();

    float core = 1.0f - smoothstep(0.0f, 0.35f, r);
    float glow = 1.0f - smoothstep(0.15f, 1.0f, r);

    float alpha = max(core * 4.0f, glow * 0.05f);

    return float4(in.color, alpha);
}

