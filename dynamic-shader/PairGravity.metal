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

kernel void computeCellMoving(
    device CellBase *attractor [[buffer(0)]],
    device CellBase *outputAttractor [[buffer(1)]],
    constant ComputeParams &params [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
  CellBase cell = attractor[id];
  device CellBase &outputCell = outputAttractor[id];

  bool leading = (id % (params.groupSize + 1) == 0);
  float3 center = float3(-1.5, 0.0, -1.0);
  float3 center2 = float3(1.5, 0.0, -1.0);

  float dt = params.elapsed * 2;

  if (leading) {

    float3 newPosition = cell.position + cell.velocity * dt;

    float3 toCenter = center - newPosition;
    float dist = length(toCenter);
    // Inverse square law with small offset to avoid division by zero
    float gravityStrength = 0.02 / (dist * dist + 0.6);
    float3 forceToCenter = normalize(toCenter) * gravityStrength;

    float3 toCenter2 = center2 - newPosition;
    float dist2 = length(toCenter2);
    float gravityStrength2 = 0.002 / (dist2 * dist2 + 0.6);
    float3 forceToCenter2 = normalize(toCenter2) * gravityStrength2;

    outputCell.position = newPosition;
    outputCell.velocity = cell.velocity + (forceToCenter + forceToCenter2) * dt;
    outputCell.color = cell.color;

  } else {
    // copy previous
    outputCell.position = outputAttractor[id - 1].position;
    outputCell.color = outputAttractor[id - 1].color;
    outputCell.velocity = outputAttractor[id - 1].velocity;
  }
}
