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
} BounceInBallVertexIn;

typedef struct {
  float4 position [[position]];
  float4 color;
} BounceInBallInOut;

typedef struct {
  float time;
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

struct IntersectionInfo {
  bool intersected;
  float3 hitPoint;
  float3 normal;      // normal at the intersection point
  float moveDistance; // distance to the intersection point
  float moveTime;     // time to move to the intersection point
};

/// Calculates the intersection of a ray with an axis-aligned cube, assuming the
/// starting point is inside the cube, detect if the ray exits the cube.
static IntersectionInfo calculateCubeIntersection(
    float3 center, float halfSize, float3 p0, float3 velocity) {
  IntersectionInfo info;
  info.intersected = false;
  info.moveDistance = 0.0;
  info.moveTime = 0.0;

  // check if p0 is inside the cube, try abs
  if (abs(p0.x - center.x) < halfSize || abs(p0.y - center.y) < halfSize ||
      abs(p0.z - center.z) < halfSize) {
    return info; // inside the cube
  }

  float timeToExit = FLT_MAX;

  // try x=1 plane
  float tX = (center.x + halfSize - p0.x) / velocity.x;
  if (tX > 0.0 && tX < timeToExit) {
    timeToExit = tX;
    info.hitPoint = p0 + velocity * tX;
    info.normal = float3(1.0, 0.0, 0.0);
    info.moveDistance = length(info.hitPoint - p0);
    info.moveTime = tX;
    info.intersected = true;
  }
  // try x=-1 plane
  tX = (center.x - halfSize - p0.x) / velocity.x;
  if (tX > 0.0 && tX < timeToExit) {
    timeToExit = tX;
    info.hitPoint = p0 + velocity * tX;
    info.normal = float3(-1.0, 0.0, 0.0);
    info.moveDistance = length(info.hitPoint - p0);
    info.moveTime = tX;
    info.intersected = true;
  }
  // try y=1 plane
  float tY = (center.y + halfSize - p0.y) / velocity.y;
  if (tY > 0.0 && tY < timeToExit) {
    timeToExit = tY;
    info.hitPoint = p0 + velocity * tY;
    info.normal = float3(0.0, 1.0, 0.0);
    info.moveDistance = length(info.hitPoint - p0);
    info.moveTime = tY;
    info.intersected = true;
  }
  // try y=-1 plane
  tY = (center.y - halfSize - p0.y) / velocity.y;
  if (tY > 0.0 && tY < timeToExit) {
    timeToExit = tY;
    info.hitPoint = p0 + velocity * tY;
    info.normal = float3(0.0, -1.0, 0.0);
    info.moveDistance = length(info.hitPoint - p0);
    info.moveTime = tY;
    info.intersected = true;
  }
  // try z=1 plane
  float tZ = (center.z + halfSize - p0.z) / velocity.z;
  if (tZ > 0.0 && tZ < timeToExit) {
    timeToExit = tZ;
    info.hitPoint = p0 + velocity * tZ;
    info.normal = float3(0.0, 0.0, 1.0);
    info.moveDistance = length(info.hitPoint - p0);
    info.moveTime = tZ;
  }
  // try z=-1 plane
  tZ = (center.z - halfSize - p0.z) / velocity.z;
  if (tZ > 0.0 && tZ < timeToExit) {
    timeToExit = tZ;
    info.hitPoint = p0 + velocity * tZ;
    info.normal = float3(0.0, 0.0, -1.0);
    info.moveDistance = length(info.hitPoint - p0);
    info.moveTime = tZ;
    info.intersected = true;
  }

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

static float cubeDistance(float3 p0, float3 p1) {
  // Calculate the distance between two points in a cube
  return max(abs(p0.x - p1.x), max(abs(p0.y - p1.y), abs(p0.z - p1.z)));
}

kernel void bounceAroundCubeComputeShader(
    device BounceInBallBase *attractor [[buffer(0)]],
    device BounceInBallBase *outputAttractor [[buffer(1)]],
    constant BounceInBallParams &params [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
  BounceInBallBase cell = attractor[id];
  device BounceInBallBase &outputCell = outputAttractor[id];

  bool leading = (id % (params.groupSize + 1) == 0);
  float3 center = float3(0.0, 0.0, -1.0);
  float r = 1.0;
  float dt = params.time * 8;
  float decay = 0.96;

  if (leading) {
    if (cubeDistance(cell.position, center) <= r) {
      // If the leading point is in the cube, leave it as is
      outputCell.position = cell.position + cell.velocity * dt;
      outputCell.color = cell.color;
      outputCell.velocity = cell.velocity;
    } else {
      float3 newPosition = cell.position + cell.velocity * dt;

      // Check if new position is moving into cube
      if (cubeDistance(newPosition, center) < r) {
        // Check if new position is moving into the cube
        IntersectionInfo info =
            calculateCubeIntersection(center, r, cell.position, cell.velocity);

        if (info.intersected &&
            info.moveDistance <= length(cell.velocity * dt)) {
          // Time to collision
          float timeToCollision = info.moveDistance / length(cell.velocity);

          // Reflect velocity(angle changes, could ESCAPE cube)
          float3 perpVelocity = dot(cell.velocity, info.normal) * info.normal;
          float3 parallelVelocity = cell.velocity - perpVelocity;
          float3 newVelocity = parallelVelocity - perpVelocity * decay;

          // Remaining time after collision
          float remainingTime = dt - timeToCollision;

          float3 nextPosition = info.hitPoint + newVelocity * remainingTime;

          // New position after bounce for the remaining time
          outputCell.position = nextPosition;
          outputCell.velocity = newVelocity;
          outputCell.color = cell.color;
        } else {
          // invalid intersection, mark as red
          outputCell.position = newPosition;
          outputCell.velocity = cell.velocity;
          outputCell.color = cell.color;
        }
      } else {
        float3 forceToCenter = normalize(center - newPosition) * 0.001;
        outputCell.position = newPosition;
        outputCell.velocity = cell.velocity + forceToCenter * dt;
        outputCell.color = cell.color;
      }
    }
  } else {
    // copy previous
    outputCell.position = outputAttractor[id - 1].position;
    outputCell.color = outputAttractor[id - 1].color;
    outputCell.velocity = outputAttractor[id - 1].velocity;
  }
}

vertex BounceInBallInOut bounceAroundCubeVertexShader(
    BounceInBallVertexIn in [[stage_in]],
    ushort amp_id [[amplification_id]],
    constant Uniforms &uniforms [[buffer(BufferIndexUniforms)]],
    constant TintUniforms &tintUniform [[buffer(BufferIndexTintUniforms)]],
    constant BounceInBallParams &params [[buffer(BufferIndexParams)]],
    const device BounceInBallBase *linesData [[buffer(BufferIndexBase)]]) {
  BounceInBallInOut out;

  UniformsPerView uniformsPerView = uniforms.perView[amp_id];
  // float3 cameraAt = uniforms.cameraPos;
  simd_float3 cameraDirection = uniforms.cameraDirection;

  int lineNumber = in.lineNumber;
  int groupNumber = in.groupNumber;
  int cellSide = in.cellSide;

  BounceInBallBase cell =
      linesData[lineNumber * (params.groupSize + 1) + groupNumber + 1];
  BounceInBallBase prevCell =
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

fragment float4 bounceAroundCubeFragmentShader(BounceInBallInOut in
                                               [[stage_in]]) {
  if (in.color.a <= 0.0) {
    discard_fragment();
  }

  return in.color;
}
