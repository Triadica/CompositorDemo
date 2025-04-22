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
  float3 position [[attribute(VertexAttributePosition)]];
  float3 color [[attribute(VertexAttributeColor)]];
  int seed [[attribute(VertexAttributeSeed)]];
} LampVertexIn;

typedef struct {
  float4 position [[position]];
  float4 color;
  float y;
} BlockInOut;

typedef struct {
  float time;
  float3 viewerPosition;
  float viewerScale;
  float viewerRotation;
} Params;

struct CellBase {
  float3 position;
  float3 color;
  float lampIdf;
  float3 velocity;
};

static float random1D(float seed) { return fract(sin(seed) * 43758.5453123); }

static float4 applyGuestureViewer(
    float4 p0,
    float3 viewerPosition,
    float viewerScale,
    float viewerRotation,
    float3 cameraAt) {

  float4 position = p0;

  // position -= cameraAt4;

  // translate
  position = position - float4(viewerPosition, 0.0);

  // rotate xz by viewerRotation
  float cosTheta = cos(viewerRotation);
  float sinTheta = sin(viewerRotation);
  float x = position.x * cosTheta - position.z * sinTheta;
  float z = position.x * sinTheta + position.z * cosTheta;
  position.x = x;
  position.z = z;

  // scale
  position *= viewerScale;

  // position += cameraAt4;

  return position;
}

kernel void blocksComputeShader(
    device CellBase *blocks [[buffer(0)]],
    device CellBase *outputLamps [[buffer(1)]],
    constant Params &params [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
  // CellBase block = blocks[id];
  // device CellBase &outputBlock = outputLamps[id];
  // float seed = fract(block.lampIdf / 10.) * 10.;
  // float speed = random1D(seed) + 0.1;
  // float dt = params.time * speed * 0.1;
  // outputBlock.position =
  //     block.position + float3(0.0, dt, 0.0) + block.velocity * dt;
  // outputBlock.color = block.color;
  // outputBlock.lampIdf = block.lampIdf;
}

vertex BlockInOut blocksVertexShader(
    LampVertexIn in [[stage_in]],
    ushort amp_id [[amplification_id]],
    constant Uniforms &uniforms [[buffer(BufferIndexUniforms)]],
    constant TintUniforms &tintUniform [[buffer(BufferIndexTintUniforms)]],
    constant Params &params [[buffer(BufferIndexParams)]],
    const device CellBase *blocksData [[buffer(BufferIndexBase)]]) {
  BlockInOut out;

  UniformsPerView uniformsPerView = uniforms.perView[amp_id];
  float3 cameraAt = uniforms.cameraPos;

  float4 position = float4(in.position + blocksData[in.seed].position, 1.0);
  float y = position.y;

  float blockDistance = distance(cameraAt, position.xyz);
  float distanceDim = 1.0 - clamp(blockDistance / 150.0, 0.0, 0.9);
  float randSeed = random1D(blocksData[in.seed].lampIdf);

  position = applyGuestureViewer(
      position,
      params.viewerPosition,
      params.viewerScale,
      params.viewerRotation,
      cameraAt);

  position.w = 1.0; // reset w to 1.0 for the projection matrix

  out.y = y;
  out.position = uniformsPerView.modelViewProjectionMatrix * position;
  out.color = float4(in.color, tintUniform.tintOpacity);
  // Premultiply color channel by alpha channel.
  out.color.rgb = out.color.rgb * out.color.a * distanceDim;

  return out;
}

fragment float4 blocksFragmentShader(BlockInOut in [[stage_in]]) {
  if (in.color.a <= 0.0) {
    discard_fragment();
  }

  float4 color = in.color;
  if (abs(fract(in.y / 2.0)) < 0.08) {
    color -= float4(0.1, 0.1, 0.1, 0.0);
  }

  return color;
}
