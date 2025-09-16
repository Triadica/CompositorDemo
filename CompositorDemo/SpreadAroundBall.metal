/*
 See the LICENSE.txt file for this sample’s licensing information.

 Abstract:
 Contains the vertex and fragment shaders for the path and tint renderers.
 */

#include <metal_stdlib>
#include <simd/simd.h>

// Include header shared between this Metal shader code and Swift/C code.
// (This needs to be imported first)
#import "ShaderTypes.h"

// executing Metal API commands.
#import "PathProperties.h"

using namespace metal;

typedef struct {
  float3 position [[attribute(0)]];
  int lineNumber [[attribute(1)]];
  int groupNumber [[attribute(2)]];
  int cellSide [[attribute(3)]];
} SpreadAroundBallVertexIn;

typedef struct {
  float4 position [[position]];
  float4 color;
} SpreadAroundBallInOut;

typedef struct {
  float time;
  int groupSize;
  float3 viewerPosition;
  float viewerScale;
  float viewerRotation;
} SpreadAroundBallParams;

struct SpreadAroundBallBase {
  float3 position;
  float3 color;
  float3 velocity;
};

struct IntersectionInfo {
  bool intersected;
  float3 intersectionPoint;
  float3 normal;
  float moveDistance;
};

/// find out the point a ray intersects with a sphere
static IntersectionInfo
    calculateSphereIntersection(float3 center, float r, float3 p0, float3 v0) {
  IntersectionInfo info;
  info.intersected = false;

  // Normalize the direction vector
  float3 rayDir = normalize(v0);

  // Vector from ray start to sphere center
  float3 startToCenter = center - p0;

  // Project startToCenter onto ray direction to find closest approach
  float projectionLength = dot(startToCenter, rayDir);

  // Find closest point on ray to sphere center
  float3 closestPoint = p0 + projectionLength * rayDir;

  // Distance from sphere center to closest point on ray
  float centerToRayDistance = length(center - closestPoint);

  if (centerToRayDistance > r) {
    // Ray misses sphere
    return info;
  }

  // Half-length of intersection chord
  float halfChord = sqrt(r * r - centerToRayDistance * centerToRayDistance);

  // Distance to first intersection point
  float t = projectionLength - halfChord;

  if (t < 0) {
    // Check second intersection point
    t = projectionLength + halfChord;
    if (t < 0) {
      // Both intersections behind ray start
      return info;
    }
  }

  info.intersected = true;
  info.moveDistance = t;
  info.intersectionPoint = p0 + rayDir * t;
  info.normal = normalize(info.intersectionPoint - center);
  return info;
}

static float4 applyGestureViewer(
    float4 p0,
    float3 viewerPosition,
    float viewerScale,
    float viewerRotation,
    float3 cameraAt) {

  float4 position = p0;

  // position -= cameraAt;

  // rotate xz by viewerRotation
  float cosTheta = cos(viewerRotation);
  float sinTheta = sin(viewerRotation);
  float x = position.x * cosTheta - position.z * sinTheta;
  float z = position.x * sinTheta + position.z * cosTheta;
  position.x = x;
  position.z = z;

  // scale
  position *= viewerScale;

  // translate
  position = position - float4(viewerPosition, 0.0);

  // position += cameraAt;

  return position;
}

kernel void spreadAroundBallComputeShader(
    device SpreadAroundBallBase *attractor [[buffer(0)]],
    device SpreadAroundBallBase *outputAttractor [[buffer(1)]],
    constant SpreadAroundBallParams &params [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
  SpreadAroundBallBase cell = attractor[id];
  device SpreadAroundBallBase &outputCell = outputAttractor[id];

  bool leading = (id % (params.groupSize + 1) == 0);
  float3 center = float3(0.0, 0.0, -1.0);
  float r = 1.6;
  float dt = params.time * 8;

  if (leading) {
    float3 newPosition = cell.position + cell.velocity * dt;
    float distanceToCenter = distance(cell.position, center);

    // Removed angular velocity component - following natural physics

    // Apply only center attraction force
    // Random drift is now initialized in Swift code
    float3 centerAttraction = normalize(center - cell.position) * 0.002;

    float3 totalForce = centerAttraction;

    if (distanceToCenter <= r) {
      // Inside sphere: apply forces and gentle damping
      outputCell.velocity = cell.velocity + totalForce * dt;
      outputCell.position = cell.position + outputCell.velocity * dt;
      outputCell.color = cell.color;
    } else {
      // Outside sphere: check for collision
      IntersectionInfo info =
          calculateSphereIntersection(center, r, cell.position, cell.velocity);

      if (info.intersected && info.moveDistance <= length(cell.velocity * dt)) {
        // Collision with sphere: slide along surface instead of bouncing
        float3 perpVelocity = dot(cell.velocity, info.normal) * info.normal;
        float3 parallelVelocity = cell.velocity - perpVelocity;

        // Convert perpendicular velocity to tangential motion along sphere
        // surface
        float3 tangentialFromPerp =
            normalize(cross(cross(info.normal, cell.velocity), info.normal));

        // New velocity: keep parallel component unchanged, convert
        // perpendicular to tangential with damping
        float3 newVelocity =
            parallelVelocity * 0.96 + tangentialFromPerp * 0.02;

        // Position particle slightly above surface to prevent penetration
        float3 surfacePosition = center + info.normal * (r + 0.001);
        outputCell.position = surfacePosition;
        outputCell.velocity = newVelocity;
        outputCell.color = cell.color;
      } else {
        // No collision: apply forces with inertial motion (no damping)
        outputCell.velocity = cell.velocity + totalForce * dt;
        outputCell.position = newPosition;
        outputCell.color = cell.color;
      }
    }
  } else {
    // Following particles: copy from previous with slight delay
    outputCell.position = outputAttractor[id - 1].position;
    outputCell.color = outputAttractor[id - 1].color;
    outputCell.velocity = outputAttractor[id - 1].velocity;
  }
}

vertex SpreadAroundBallInOut spreadAroundBallVertexShader(
    SpreadAroundBallVertexIn in [[stage_in]],
    ushort amp_id [[amplification_id]],
    constant Uniforms &uniforms [[buffer(BufferIndexUniforms)]],
    constant TintUniforms &tintUniform [[buffer(BufferIndexTintUniforms)]],
    constant SpreadAroundBallParams &params [[buffer(BufferIndexParams)]],
    const device SpreadAroundBallBase *linesData [[buffer(BufferIndexBase)]]) {
  SpreadAroundBallInOut out;

  UniformsPerView uniformsPerView = uniforms.perView[amp_id];
  // float3 cameraAt = uniforms.cameraPos;
  simd_float3 cameraDirection = uniforms.cameraDirection;

  int lineNumber = in.lineNumber;
  int groupNumber = in.groupNumber;
  int cellSide = in.cellSide;

  SpreadAroundBallBase cell =
      linesData[lineNumber * (params.groupSize + 1) + groupNumber + 1];
  SpreadAroundBallBase prevCell =
      linesData[lineNumber * (params.groupSize + 1) + groupNumber];

  float3 direction = cell.position - prevCell.position;
  float3 brush = normalize(cross(direction, cameraDirection)) * 0.001;

  float4 position = float4(0., 0., 0., 1.0);
  if (cellSide == 0) {
    position = float4(prevCell.position + brush, 1.0);
  } else if (cellSide == 1) {
    position = float4(prevCell.position - brush, 1.0);
  } else if (cellSide == 2) {
    position = float4(cell.position + brush, 1.0);
  } else if (cellSide == 3) {
    position = float4(cell.position - brush, 1.0);
  }

  position = applyGestureViewer(
      position,
      params.viewerPosition,
      params.viewerScale,
      params.viewerRotation,
      uniforms.cameraPos);

  position.w = 1.0; // need to be 1.0 for perspective projection

  out.position = uniformsPerView.modelViewProjectionMatrix * position;
  out.color = float4(cell.color, 1.0);

  return out;
}

fragment float4 spreadAroundBallFragmentShader(SpreadAroundBallInOut in
                                               [[stage_in]]) {
  if (in.color.a <= 0.0) {
    discard_fragment();
  }

  return in.color;
}
