
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

} LampInOut;

typedef struct {
  float3 viewerPosition;
  float time;
  float viewerScale;
  float viewerRotation;
} Params;

struct RaindropBase {
  float3 position;
  float3 color;
  float raindropId;
  float3 velocity;
  float groundTime; // Time staying on the ground
    bool isOnGround;  // Whether on the ground
};

static float random1D(float seed) { return fract(sin(seed) * 43758.5453123); }

static float4 applyGestureViewerOnScene(
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

kernel void rainComputeShader(
    device RaindropBase *raindrops [[buffer(0)]],
    device RaindropBase *outputRaindrops [[buffer(1)]],
    constant Params &params [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {

  RaindropBase raindrop = raindrops[id];
  device RaindropBase &outputRaindrop = outputRaindrops[id];

  float seed = raindrop.raindropId + float(id) * 0.1;

  // Use a more stable time delta
  float dt = max(abs(params.time), 0.016); // At least 16ms (60fps)
  dt = min(dt, 0.1);                       // Cap at 100ms to prevent huge jumps

  // Ground level and reset height
  const float groundLevel = -0.5;
  const float resetHeight = 8.0; // Height to reset to

  // Copy basic properties
  outputRaindrop.color = raindrop.color;
  outputRaindrop.raindropId = raindrop.raindropId;

  // Ensure consistent downward velocity for falling raindrops
  outputRaindrop.velocity = raindrop.velocity;

  // If velocity is too small (possibly uninitialized), set default falling
  // velocity
  if (abs(outputRaindrop.velocity.y) < 0.5) {
    outputRaindrop.velocity.y = -(1.5 + random1D(seed) * 0.5); // -1.5 to -2.0
  }

  // Normal falling motion
  float3 newPosition = raindrop.position + outputRaindrop.velocity * dt;

  // Check if reached ground - if so, immediately reset to high altitude
  if (newPosition.y <= groundLevel) {
    // Use time-based seed for better randomness
    float timeSeed = seed + float(id) * 0.234 + params.time * 0.01;

    // Reset position to random high altitude
    outputRaindrop.position.y = resetHeight + random1D(timeSeed * 1.23) * 4.0;
    outputRaindrop.position.x = (random1D(timeSeed * 2.47) - 0.5) * 50.0;
    outputRaindrop.position.z = (random1D(timeSeed * 3.71) - 0.5) * 50.0;

    // Reset velocity for falling with new random values
    outputRaindrop.velocity = float3(
        (random1D(timeSeed * 4.13) - 0.5) * 0.4,  // Horizontal drift
        -(1.2 + random1D(timeSeed * 5.29) * 0.6), // Downward speed -1.2 to -1.8
        (random1D(timeSeed * 6.83) - 0.5) * 0.4   // Horizontal drift
    );

    // Clear ground-related states (not used anymore but keep for compatibility)
    outputRaindrop.isOnGround = false;
    outputRaindrop.groundTime = 0.0;
  } else {
    // Normal falling - just update position
    outputRaindrop.position = newPosition;
    outputRaindrop.isOnGround = false;
    outputRaindrop.groundTime = 0.0;
  }
}

vertex LampInOut rainVertexShader(
    LampVertexIn in [[stage_in]],
    ushort amp_id [[amplification_id]],
    constant Uniforms &uniforms [[buffer(BufferIndexUniforms)]],
    constant TintUniforms &tintUniform [[buffer(BufferIndexTintUniforms)]],
    constant Params &params [[buffer(BufferIndexParams)]],
    const device RaindropBase *raindropData [[buffer(BufferIndexBase)]]) {
  LampInOut out;

  UniformsPerView uniformsPerView = uniforms.perView[amp_id];
  float3 cameraAt = uniforms.cameraPos;

  // Use position directly from compute buffer, no additional calculation needed
  float4 position = float4(in.position + raindropData[in.seed].position, 1.0);

  float raindropDistance = distance(cameraAt, position.xyz);
  float distanceDim = 1.0 - clamp(raindropDistance / 40.0, 0.0, 0.8);

  position = applyGestureViewerOnScene(
      position,
      params.viewerPosition,
      params.viewerScale,
      params.viewerRotation,
      uniforms.cameraPos);

  position.w = 1;

  out.position = uniformsPerView.modelViewProjectionMatrix * position;
  out.color = float4(in.color, tintUniform.tintOpacity);
  // Premultiply color channel by alpha channel.
  out.color.rgb = out.color.rgb * out.color.a * distanceDim;

  return out;
}

fragment float4 rainFragmentShader(LampInOut in [[stage_in]]) {
  if (in.color.a <= 0.0) {
    discard_fragment();
  }

  return in.color;
}
