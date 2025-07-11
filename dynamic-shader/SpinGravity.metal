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
} BounceInBallParams;

struct BounceInBallBase {
  float3 position;
  float3 color;
  float3 velocity;
};

kernel void multiGravityComputeShader(
    device BounceInBallBase *attractor [[buffer(0)]],
    device BounceInBallBase *outputAttractor [[buffer(1)]],
    constant BounceInBallParams &params [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
  BounceInBallBase cell = attractor[id];
  device BounceInBallBase &outputCell = outputAttractor[id];

  bool leading = (id % (params.groupSize + 1) == 0);
  float3 center = float3(-1.5, 0.0, -1.0);

  float angle = params.time * 0.1;
  float r = 0.4;
  float x = cos(angle) * r - sin(angle) * r;
  float z = sin(angle) * r + cos(angle) * r;
  center += float3(x, 0.0, z);

  float dt = params.elapsed * 2;

  if (leading) {

    float3 newPosition = cell.position + cell.velocity * dt;

    float3 toCenter = center - newPosition;
    float dist = length(toCenter);
    // Inverse square law with small offset to avoid division by zero
    float gravityStrength = 0.04 / (dist * dist + 0.6);
    float3 forceToCenter = normalize(toCenter) * gravityStrength;

    float3 dampling = -cell.velocity * 0.02;

    outputCell.position = newPosition;
    outputCell.velocity = cell.velocity + (forceToCenter + dampling) * dt;
    outputCell.color = cell.color;

  } else {
    // copy previous
    outputCell.position = outputAttractor[id - 1].position;
    outputCell.color = outputAttractor[id - 1].color;
    outputCell.velocity = outputAttractor[id - 1].velocity;
  }
}
