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

  if (leading) {
    float dt = params.elapsed * 2;

    // 4D Tesseract rotation projected to 3D space
    float3 pos = cell.position;
    float t = params.time * 0.2;

    // Create circular ring motion in XY plane with Z perturbations
    float radius = 3.0 + sin(t * 0.3) * 0.5; // Varying ring radius
    float angle = t * 0.8 + id * 0.1; // Different phase for each particle

    // Base ring position
    float3 ringCenter = float3(cos(angle) * radius, sin(angle) * radius, 0.0);

    // Add Z-direction oscillations
    float zOscillation =
        sin(t * 1.2 + id * 0.2) * 0.8 + cos(t * 0.7 + angle) * 0.4;
    ringCenter.z = zOscillation;

    // Add radial perturbations
    float radialNoise = sin(t * 2.0 + id * 0.5) * 0.3;
    float tangentialNoise = cos(t * 1.5 + id * 0.3) * 0.2;

    // Apply perturbations
    float3 radialDir = normalize(float3(cos(angle), sin(angle), 0.0));
    float3 tangentialDir = float3(-sin(angle), cos(angle), 0.0);
    float3 verticalNoise = float3(0.0, 0.0, sin(t * 3.0 + id * 0.8) * 0.15);

    float3 target = ringCenter + radialDir * radialNoise +
                    tangentialDir * tangentialNoise + verticalNoise;

    // Smooth movement towards target
    float3 force = (target - pos) * 0.05;

    // Add some swirling motion
    float3 swirl =
        float3(-sin(angle + t), cos(angle + t), sin(t * 2.0 + angle) * 0.3) *
        0.02;
    force += swirl;

    // Boundary constraint
    float maxRadius = 6.0;
    float currentRadius = length(pos);
    if (currentRadius > maxRadius) {
      force += -normalize(pos) * (currentRadius - maxRadius) * 0.2;
    }

    float3 newVelocity = cell.velocity + force * dt;
    newVelocity *= 0.92; // Damping

    float3 newPosition = cell.position + newVelocity * dt;
    float3 deltaV = force * dt * 0.5;

    // write to cell
    outputCell.position = newPosition;
    outputCell.velocity = cell.velocity + deltaV;
    outputCell.color = cell.color;

  } else {
    // move along leading cell
    outputCell.position = outputAttractor[id - 1].position;
    outputCell.color = outputAttractor[id - 1].color;
    // outputCell.velocity = outputAttractor[id - 1].velocity;
  }
}
