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

  // position -= cameraAt4;

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

  // position += cameraAt4;

  return position;
}

// a Metal function of fourwing
static float3 fourwingLineIteration(float3 p, float dt) {
  float a = 0.2;
  float b = 0.01;
  float c = -0.4;
  float x = p.x;
  float y = p.y;
  float z = p.z;
  float dx = a * x + y * z;
  float dy = b * x + c * y - x * z;
  float dz = -z - x * y;
  float3 d = float3(dx, dy, dz) * dt;
  return p + d;
}

// a Metal function of lorenz
float3 lorenzLineIteration(float3 p, float dt) {
  float tau = 10.0;
  float rou = 28.0;
  float beta = 8.0 / 3.0;

  float dx = tau * (p.y - p.x);
  float dy = p.x * (rou - p.z) - p.y;
  float dz = p.x * p.y - beta * p.z;
  float3 d = float3(dx, dy, dz) * dt;
  return p + d;
}

kernel void attractorComputeShader(
    device BounceInBallBase *attractor [[buffer(0)]],
    device BounceInBallBase *outputAttractor [[buffer(1)]],
    constant BounceInBallParams &params [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
  BounceInBallBase lamp = attractor[id];
  device BounceInBallBase &outputCell = outputAttractor[id];

  bool leading = (id % (params.groupSize + 1) == 0);

  if (leading) {
    float dt = params.time * 2;
    outputCell.position = fourwingLineIteration(lamp.position, dt);
    // outputCell.position = lorenzLineIteration(outputCell.position, dt);
    outputCell.color = lamp.color;
  } else {
    // copy previous
    outputCell.position = outputAttractor[id - 1].position;
    outputCell.color = outputAttractor[id - 1].color;
  }
}

vertex BounceInBallInOut attractorVertexShader(
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

fragment float4 attractorFragmentShader(BounceInBallInOut in [[stage_in]]) {
  if (in.color.a <= 0.0) {
    discard_fragment();
  }

  return in.color;
}
