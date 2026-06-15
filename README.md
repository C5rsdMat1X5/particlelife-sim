# Particle Life Sim

A real-time particle life simulation running entirely on the GPU via Apple Metal. 131 072 particles self-organize into emergent structures driven by per-type attraction/repulsion rules generated at random each run.

![particles forming clusters with colored glow](/img/placeholder.png)

---

## What it does

Each particle belongs to one of **16 types**. A randomly generated **16×16 interaction matrix** defines whether any two types attract or repel each other, and at what distances. From these simple rules complex behaviour emerges — spinning clusters, predator-prey chains, crystalline lattices — different every time.

The physics runs in a **spatial hash grid** (512×512 cells, 8-cell search radius) so force lookups stay O(N) instead of O(N²), making 131 K particles feasible at interactive frame rates.

Rendering uses **additive alpha blending**: a tight semi-transparent core shows each particle's type color, surrounded by a soft yellow halo that accumulates brightness wherever particles cluster, giving a natural bloom effect without a separate post-process pass.

---

## Requirements

| Dependency | How to get |
|---|---|
| macOS 13+ | Metal 3 / Apple Silicon or Intel Mac with Metal support |
| Xcode Command Line Tools | `xcode-select --install` |
| GLFW 3 | `brew install glfw` |
| pkg-config | `brew install pkg-config` |

---

## Build & run

```bash
bash build.sh
./build/circles
```

The shader source (`src/shaders.metal`) is compiled at **runtime** from the working directory, so run the binary from the repo root.

---

## Controls

| Input | Action |
|---|---|
| Scroll wheel | Zoom in / out (anchored to cursor) |
| Left drag | Pan |
| `R` | Reset camera |
| `↑` / `↓` | Speed up / slow down simulation |
| `0` | Reset simulation speed to 1× |
| `Escape` | Quit |

The camera uses **exponential smoothing** (zoom in log-space) so all movement feels fluid even at large zoom deltas.

---

## Tuning

All simulation parameters live at the top of `src/main.cpp`:

```cpp
static const uint32_t N          = 1 << 17;   // particle count (power of 2)
static const uint32_t NUM_TYPES  = 16;         // number of particle types
static const float    G          = 5e-7f;      // attraction strength
static const float    REPEL_FORCE= 0.05f;      // short-range repulsion
static const float    DRAG       = 0.95f;      // velocity damping per step
static const float    RAND_SCALE = 4.0f;       // interaction matrix range [-R, R]
static const uint32_t GRID_DIM   = 512;        // spatial hash grid resolution
static const int      SEARCH_RAD = 8;          // neighbour search radius (cells)
```

Visual parameters live in `src/shaders.metal` — point size, glow shape and halo color are all in the `particle_vert` / `particle_frag` functions.

---

## Architecture

```
src/
├── main.cpp          – simulation setup, GLFW window, input, main loop
├── metal_sim.mm      – Metal device, compute pipelines, render pipeline, frame submission
├── metal_sim.hpp     – shared types (Particle, Camera) and C API
└── shaders.metal     – GPU kernels (grid sort + force) and render shaders (vertex + fragment)
```

**Frame pipeline per tick:**

1. `clear_cells` — zero the 512² cell count buffer
2. `count_cells` — each particle atomically increments its cell's counter
3. CPU prefix-sum — builds `starts[]` offsets (single-threaded, ~262 K adds, negligible)
4. `scatter_particles` — copies particles into a sorted buffer by cell
5. `grid_force` — each particle reads its 17×17 cell neighbourhood, accumulates forces, integrates velocity + position (Euler)
6. Render pass — draw particles as point sprites with additive blend directly to the CAMetalLayer drawable

---

## License

MIT
