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

// static float random1D(float seed) { return fract(sin(seed) * 43758.5453123);
// }

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

kernel void bounceInBallComputeShader(
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
  float decay = 0.94;

  if (leading) {
    if (distance(cell.position, center) > r) {
      float3 directionToCenter = normalize(center - cell.position);
      outputCell.position = cell.position + directionToCenter * r;
      outputCell.color = cell.color;
      outputCell.velocity = cell.velocity;
    } else {
      float3 newPosition = cell.position + cell.velocity * dt;

      // Check if new position is outside the sphere
      if (distance(newPosition, center) > r) {
        // Calculate intersection point with sphere
        float3 toCenter = normalize(center - cell.position);
        float3 normal = normalize(newPosition - center);

        // Reflect velocity off the sphere surface
        float3 parallelVelocity =
            cell.velocity - dot(cell.velocity, normal) * normal;
        float3 perpVelocity = dot(cell.velocity, normal) * normal;
        outputCell.velocity = parallelVelocity - perpVelocity * decay;

        // just stay on the sphere surface
        outputCell.position = cell.position;
      } else {
        float3 acc = float3(0.0, -0.002, 0.0);
        outputCell.position = newPosition;
        outputCell.velocity = cell.velocity + acc * dt;
      }

      outputCell.color = cell.color;
    }
  } else {
    // copy previous
    outputCell.position = outputAttractor[id - 1].position;
    outputCell.color = outputAttractor[id - 1].color;
    outputCell.velocity = outputAttractor[id - 1].velocity;
  }
}

vertex BounceInBallInOut bounceInBallVertexShader(
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

fragment float4 bounceInBallFragmentShader(BounceInBallInOut in [[stage_in]]) {
  if (in.color.a <= 0.0) {
    discard_fragment();
  }

  return in.color;
}
