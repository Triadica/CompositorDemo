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
    // Create multiple 3D vortices pattern
    float t = params.time * 0.3;
    float3 pos = cell.position;
    float3 vel = cell.velocity;

    // Define multiple vortex centers
    const int numVortices = 4;
    float3 vortexCenters[numVortices] = {
        float3(1.0, 0.5, 0.0),
        float3(-1.0, -0.5, 0.0),
        float3(0.0, 0.0, 1.0),
        float3(0.0, 0.7, -1.0)};

    float vortexStrengths[numVortices] = {
        0.8 + 0.2 * sin(t * 0.7),
        0.7 + 0.3 * sin(t * 0.5),
        0.9 + 0.1 * sin(t * 0.6),
        0.6 + 0.4 * sin(t * 0.4)};

    // Calculate combined influence from all vortices
    float3 acceleration = float3(0.0);

    for (int i = 0; i < numVortices; i++) {
      // Animate vortex centers
      float3 center = vortexCenters[i];
      center.x += sin(t * 0.3 + i * 1.5) * 0.4;
      center.z += cos(t * 0.4 + i * 1.2) * 0.4;

      // Vector from vortex center to particle
      float3 toParticle = pos - center;
      float dist = length(toParticle);

      // Skip if too close to avoid extreme forces
      if (dist < 0.1)
        continue;

      // Normalized direction vector
      float3 dir = normalize(toParticle);

      // Cross product creates rotation around the vortex center
      float3 rotationForce = cross(float3(0, 1, 0), dir) * vortexStrengths[i];

      // Add some vertical movement based on distance
      float verticalForce = sin(dist * 5.0 + t) * 0.3;

      // Combine forces with distance attenuation
      float attenuatedStrength = 1.0 / (1.0 + dist * 0.8);
      acceleration +=
          (rotationForce + float3(0, verticalForce, 0)) * attenuatedStrength;
    }

    // Add some turbulence
    float3 noise =
        float3(
            sin(pos.y * 4.3 + t * 1.1) * cos(pos.z * 3.7 + t * 0.8),
            sin(pos.z * 4.1 + t * 0.9) * cos(pos.x * 3.5 + t * 1.0),
            sin(pos.x * 4.7 + t * 1.2) * cos(pos.y * 3.8 + t * 0.7)) *
        0.15;

    acceleration += noise;

    // Apply soft boundary to keep particles within sphere
    float dist = length(pos);
    if (dist > 2.8) {
      acceleration += normalize(-pos) * (dist - 2.8) * 2.2;
    }

    // Update velocity with damping
    float damping = 0.94;
    float3 deltaV = acceleration * dt;
    float3 newVelocity = vel * damping + deltaV;

    // Limit maximum velocity
    float maxSpeed = 2.5;
    if (length(newVelocity) > maxSpeed) {
      newVelocity = normalize(newVelocity) * maxSpeed;
    }

    // Update position
    float3 newPosition = pos + newVelocity * dt;

    // write to cell
    outputCell.position = newPosition;
    outputCell.velocity = cell.velocity + deltaV;
    outputCell.color = cell.color;
  }
}
