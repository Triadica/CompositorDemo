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

    // I don't know, Claude generated this...

    // 4D Tesseract rotation projected to 3D space
    float3 pos = cell.position;
    float t = params.time * 0.2;

    // Create 4D position by adding a fourth dimension
    float4 pos4D =
        float4(pos.x, pos.y, pos.z, sin(t + length(pos) * 0.5) * 2.0);

    // 4D rotation matrices (XY, XZ, XW, YZ, YW, ZW planes)
    float angle1 = t * 0.3;
    float angle2 = t * 0.7;
    float angle3 = t * 0.5;

    // 4D to 3D projection matrix with perspective
    float w_distance = 5.0 + pos4D.w;
    float3 projected = pos4D.xyz / (1.0 + pos4D.w / w_distance);

    // Klein bottle parametric surface in 4D projected to 3D
    float u = atan2(pos.y, pos.x) + t * 0.1;
    float v = length(pos.xy) * 0.5 + t * 0.2;

    float kleinX = (2.5 + 1.5 * cos(v)) * cos(u);
    float kleinY = (2.5 + 1.5 * cos(v)) * sin(u);
    float kleinZ = -2.5 * sin(v);
    float kleinW = 1.5 * sin(v) * cos(v / 2.0);

    float4 kleinPoint = float4(kleinX, kleinY, kleinZ, kleinW);
    float3 kleinProjected = kleinPoint.xyz / (1.0 + kleinPoint.w / 6.0);

    // Hopf fibration - beautiful 4D to 3D mapping
    float4 hopfPoint =
        normalize(float4(pos.x, pos.y, pos.z, sin(t + pos.x * 0.5)));
    float3 hopfProjected = float3(
        2.0 * (hopfPoint.x * hopfPoint.z + hopfPoint.y * hopfPoint.w),
        2.0 * (hopfPoint.y * hopfPoint.z - hopfPoint.x * hopfPoint.w),
        hopfPoint.x * hopfPoint.x + hopfPoint.y * hopfPoint.y -
            hopfPoint.z * hopfPoint.z - hopfPoint.w * hopfPoint.w);

    // 4D hyperknot projected to 3D
    float s = t + length(pos) * 0.1;
    float4 hyperknot = float4(
                           sin(2.0 * s),
                           sin(3.0 * s) * cos(s),
                           cos(2.0 * s) * sin(s),
                           cos(3.0 * s)) *
                       2.0;
    float3 hyperknotProjected = hyperknot.xyz / (1.0 + hyperknot.w / 4.0);

    // 4D Clifford torus
    float4 clifford = float4(
                          cos(s) * cos(2.0 * s),
                          sin(s) * cos(2.0 * s),
                          cos(s) * sin(2.0 * s),
                          sin(s) * sin(2.0 * s)) *
                      1.5;
    float3 cliffordProjected = clifford.xyz * (2.0 / (2.0 + clifford.w));

    // Combine different 4D projections with time-varying weights
    float w1 = (sin(t * 0.2) + 1.0) * 0.5;
    float w2 = (cos(t * 0.3) + 1.0) * 0.5;
    float w3 = (sin(t * 0.4 + 1.0) + 1.0) * 0.5;
    float w4 = (cos(t * 0.1 + 2.0) + 1.0) * 0.5;

    float3 target = kleinProjected * w1 + hopfProjected * w2 +
                    hyperknotProjected * w3 + cliffordProjected * w4;

    float3 force4D = (target - pos) * 0.03;

    // Add 4D space "gravity" effect
    float4 center4D = float4(0, 0, 0, cos(t * 0.1) * 2.0);
    float4 current4D = float4(pos, sin(t + pos.x) * 1.5);
    float4 gravity4D = (center4D - current4D) * 0.01;
    float3 gravityProjected = gravity4D.xyz / (1.0 + abs(gravity4D.w) / 3.0);

    float3 combinedForce = force4D + gravityProjected;

    // Boundary constraint in 4D projected space
    float maxRadius = 5.0;
    float currentRadius = length(pos);
    if (currentRadius > maxRadius) {
      combinedForce += -normalize(pos) * (currentRadius - maxRadius) * 0.3;
    }

    float3 newVelocity = cell.velocity + combinedForce * dt;
    newVelocity *= 0.95; // Damping

    float3 newPosition = cell.position + newVelocity * dt;
    float3 deltaV = combinedForce * dt * 0.3;
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
