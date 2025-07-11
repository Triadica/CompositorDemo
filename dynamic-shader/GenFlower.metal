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

    // Create flower petal pattern in XY plane with subtle Z movements
    float3 pos = cell.position;
    float t = params.time * 0.15;

    // Petal parameters
    int petalCount = 5; // Number of petals
    float baseRadius = 2.5;
    float petalLength = 1.8;

    // Individual particle phase
    float particlePhase = id * 0.08;
    float globalAngle = t * 0.6 + particlePhase;

    // Create petal shape using rose curve: r = a * cos(k * θ)
    float petalAngle = globalAngle * petalCount;
    float petalRadius = baseRadius + petalLength * cos(petalAngle);

    // Add secondary harmonics for more complex petal shapes
    petalRadius += 0.4 * cos(petalAngle * 2.0) + 0.2 * sin(petalAngle * 3.0);

    // Ensure minimum radius to avoid center clustering
    petalRadius = max(petalRadius, 0.5);

    // Convert to cartesian coordinates
    float3 target = float3(
        cos(globalAngle) * petalRadius, sin(globalAngle) * petalRadius, 0.0);

    // Add gentle Z oscillations synchronized with petal motion
    target.z = sin(petalAngle * 0.5 + t * 2.0) * 0.3 +
               cos(globalAngle * 2.0 + t) * 0.15;

    // Add fine perturbations for organic feel
    float noise = sin(t * 4.0 + id * 0.4) * 0.1;
    float3 perturbation = float3(
        sin(globalAngle * 7.0 + t * 3.0) * noise,
        cos(globalAngle * 7.0 + t * 3.0) * noise,
        sin(t * 5.0 + id * 0.6) * 0.05);

    target += perturbation;

    // Smooth movement towards target
    float3 force = (target - pos) * 0.04;

    // Add rotational flow
    float3 tangential = float3(-sin(globalAngle), cos(globalAngle), 0.0);
    force += tangential * 0.015 * petalRadius / baseRadius;

    float3 newVelocity = cell.velocity + force * dt;
    newVelocity *= 0.88; // Damping

    float3 newPosition = cell.position + newVelocity * dt;
    float3 deltaV = force * dt * 0.6;

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
