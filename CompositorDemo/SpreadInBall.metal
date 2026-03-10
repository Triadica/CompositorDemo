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
} SpreadInBallVertexIn;

typedef struct {
  float4 position [[position]];
  float4 color;
} SpreadInBallInOut;

typedef struct {
  float time;
  int groupSize;
  float3 viewerPosition;
  float viewerScale;
  float viewerRotation;
} SpreadInBallParams;

struct SpreadInBallBase {
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
__attribute__((unused)) static IntersectionInfo
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

kernel void spreadInBallComputeShader(
    device SpreadInBallBase *attractor [[buffer(0)]],
    device SpreadInBallBase *outputAttractor [[buffer(1)]],
    constant SpreadInBallParams &params [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
  SpreadInBallBase cell = attractor[id];
  device SpreadInBallBase &outputCell = outputAttractor[id];

  bool leading = (id % (params.groupSize + 1) == 0);
  float3 center = float3(0.0, 0.0, -1.0);
  float r = 1.0; // Collision sphere radius = 1.0m (diameter = 2.0m)
  float dt = params.time * 8;

  if (leading) {
    // Removed angular velocity component - following natural physics

    // Add gravity force (slight downward acceleration)
    float3 gravity = float3(0.0, -0.0002, 0.0);

    // Remove center attraction for InBall mode - particles move freely inside
    float3 totalForce = gravity;

    // Calculate new position
    float3 newVelocity = cell.velocity + totalForce * dt;
    float3 newPosition = cell.position + newVelocity * dt;
    float newDistanceToCenter = distance(newPosition, center);

    if (newDistanceToCenter >= r) {
      // Particle would exit sphere: handle internal collision
      float3 directionFromCenter = normalize(newPosition - center);
      float3 normal =
          -directionFromCenter; // Normal points inward for internal collision

      // Calculate intersection point on sphere surface
      float3 intersectionPoint = center + directionFromCenter * r;

      // Reflect velocity off internal sphere surface
      float3 reflectedVelocity =
          newVelocity - 2.0 * dot(newVelocity, normal) * normal;

      // Apply damping to reflected velocity
      reflectedVelocity *= 0.94;

      // Position particle slightly inside surface at intersection point
      float3 correctedPosition = intersectionPoint + normal * 0.01;

      outputCell.position = correctedPosition;
      outputCell.velocity = reflectedVelocity;
      outputCell.color = cell.color;
    } else {
      // Particle stays inside sphere: normal movement
      outputCell.velocity = newVelocity;
      outputCell.position = newPosition;
      outputCell.color = cell.color;
    }
  } else {
    // Following particles: copy from previous with slight delay
    outputCell.position = outputAttractor[id - 1].position;
    outputCell.color = outputAttractor[id - 1].color;
    outputCell.velocity = outputAttractor[id - 1].velocity;
  }
}

vertex SpreadInBallInOut spreadInBallVertexShader(
    SpreadInBallVertexIn in [[stage_in]],
    ushort amp_id [[amplification_id]],
    constant Uniforms &uniforms [[buffer(BufferIndexUniforms)]],
    constant TintUniforms &tintUniform [[buffer(BufferIndexTintUniforms)]],
    constant SpreadInBallParams &params [[buffer(BufferIndexParams)]],
    const device SpreadInBallBase *linesData [[buffer(BufferIndexBase)]]) {
  SpreadInBallInOut out;

  UniformsPerView uniformsPerView = uniforms.perView[amp_id];
  // float3 cameraAt = uniforms.cameraPos;
  simd_float3 cameraDirection = uniforms.cameraDirection;

  int lineNumber = in.lineNumber;
  int groupNumber = in.groupNumber;
  int cellSide = in.cellSide;

  SpreadInBallBase cell =
      linesData[lineNumber * (params.groupSize + 1) + groupNumber + 1];
  SpreadInBallBase prevCell =
      linesData[lineNumber * (params.groupSize + 1) + groupNumber];

  float3 direction = cell.position - prevCell.position;
  float3 brush = normalize(cross(direction, cameraDirection)) * 0.002;

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

fragment float4 spreadInBallFragmentShader(SpreadInBallInOut in [[stage_in]]) {
  if (in.color.a <= 0.0) {
    discard_fragment();
  }

  return in.color;
}
