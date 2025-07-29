/*
 This file provides a compute function that can be dyanamically loaded
 */

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

typedef struct {
  float time;
  float elapsed;
  int groupSize;
  float3 viewerPosition;
  float viewerScale;
  float viewerRotation;
} ComputeParams;

struct CellBase {
  float3 position;
  float3 color;
  float3 velocity;
};

/// some logic to move cells with compute shader,
/// cells are placed like sphere with fibonacci grid.
kernel void computeCellMoving(
    device CellBase *attractor [[buffer(0)]],
    device CellBase *outputAttractor [[buffer(1)]],
    constant ComputeParams &params [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
  CellBase cell = attractor[id];
  device CellBase &outputCell = outputAttractor[id];
  bool leading = (id % (params.groupSize + 1) == 0);

  if (!leading) {
    // move along leading cell
    outputCell.position = outputAttractor[id - 1].position;
    outputCell.color = outputAttractor[id - 1].color;
    // outputCell.velocity = outputAttractor[id - 1].velocity;
  } else {
    // The cells are initially positioned on a Fibonacci sphere, with adjacent
    // cells having similar colors. The goal is to create a dynamic, wave-like
    // motion, inspired by ocean waves, while keeping the movement within a
    // confined spherical area. The motion should preserve the initial color
    // coherence, ensuring that neighboring cells move together and maintain
    // their similar colors. Avoid collapsing all cells into a single point.
    float dt = params.elapsed * 2;

    // Simplified, faster "Oceanic Waves on a Sphere" pattern
    float t = params.time * 0.8; // Faster time scale
    float3 pos = cell.position;

    // Reduced to two wave layers for simplicity
    float3 displacement = float3(0.0);
    float amplitude = 0.2; // Slightly increased amplitude
    float frequency = 2.5; // Adjusted frequency

    // Wave 1: Primary wave
    float wave1 = sin(dot(normalize(float3(1, 0.5, 0)), pos) * frequency + t);
    displacement +=
        normalize(cross(pos, float3(1, 0.5, 0))) * wave1 * amplitude;

    // Wave 2: Secondary wave
    float wave2 =
        cos(dot(normalize(float3(0, 0, 1)), pos) * frequency * 1.7 + t * 1.5);
    displacement +=
        normalize(cross(pos, float3(0, 0, 1))) * wave2 * amplitude * 0.6;

    // Combine position with wave displacements
    float3 displacedPos = pos + displacement;

    // Keep particles on sphere surface
    float3 newPosition = normalize(displacedPos) * length(cell.position);

    // Faster transition (0.7 instead of 0.4)
    newPosition = mix(pos, newPosition, dt * 0.7);

    // Calculate velocity change
    float3 deltaV = (newPosition - pos) / dt;

    // write to cell
    outputCell.position = newPosition;
    outputCell.velocity = cell.velocity + deltaV;
    outputCell.color = cell.color;
  }
}
