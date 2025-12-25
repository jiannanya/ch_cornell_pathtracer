# Cornell Box Path Tracer (C++)

A small CPU path tracer that renders a Cornell Box–style scene to a **PPM** image.

This project is intentionally minimal: everything lives in a single file, [`pathtracer.cpp`](pathtracer.cpp).

## Result

![`result`](./cornell_box_ggx_pt.png)

## Features

- **Unidirectional path tracing** with throughput accumulation
- **Russian Roulette** termination (starts after a few bounces)
- **BVH acceleration** (templated BVH, SAH-like split search)
- **Multithreaded tile renderer** using `std::thread` + atomics
- Materials:
  - Lambertian diffuse
  - Diffuse area light (`DiffuseLight`)
  - Simple PBR-style material (`PBRMaterial`)
  - GGX rough metal (`GGXMetal`) with importance sampling
- Output:
  - Linear HDR accumulation per pixel
  - Optional **Reinhard** tonemapping
  - **Gamma 2.2** correction

## Output

By default, the program writes:

- `cornell_box_ggx_pt.ppm`

PPM is a simple text image format. Many viewers can open it directly; otherwise you can convert it.

## Build

```bat
clang++ -std=c++17 -O3 -DNDEBUG pathtracer.cpp -o pathtracer.exe
```

This produces `pathtracer.exe`.

Run this inside a **Developer Command Prompt** (so MSVC libraries are available):

Notes:
- You can add `-march=native` for extra performance on your CPU.

## Run

```bat
pathtracer.exe
```

You’ll see progress in the terminal (tile-based percentage + elapsed time). When finished, the PPM is written to the working directory.

## Tuning Quality / Performance

Open [`pathtracer.cpp`](pathtracer.cpp) and modify the constants in `main()`:

- `image_width` / `image_height`: output resolution
- `samples_per_pixel`: main quality knob (noise decreases ~ $1/\sqrt{\text{spp}}$)
- `max_depth`: max bounces

The render call at the end of `main()` is:

```cpp
render_to_ppm(world, cam,
              image_width, image_height,
              samples_per_pixel, max_depth,
              "cornell_box_ggx_pt.ppm",
              1.0,  // exposure
              true  // enable_reinhard_tonemap
);
```

- **Exposure** scales the final radiance before tonemapping/gamma.
- **Reinhard tonemapping** helps prevent highlight blowout for bright lights.

## Scene Description (Current Defaults)

- Cornell Box walls (red/green/white)
- A ceiling area light (rectangle)
- Three spheres inside the box:
  - a metallic PBR sphere
  - a diffuse PBR sphere
  - a GGX rough metal sphere

## Notes / Caveats

- The environment is treated as **black** (no sky); rays that miss the scene contribute nothing.
- This is a learning-focused renderer; it does not aim to match a reference implementation pixel-for-pixel.

