/*
See the LICENSE.txt file for this sample's licensing information.

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
  float height [[attribute(3)]];
  float2 uv [[attribute(4)]];
} ImageVertexIn;

typedef struct {
  float4 position [[position]];
  float4 color;
  float3 originalPosition;
  float height;
  float2 uv;
} ImageInOut;

typedef struct {
  float time;
  float3 viewerPosition;
  float viewerScale;
  int itemsCount;
} Params;

struct CellBase {
  float3 position;
  float3 color;
  float lampIdf;
  bool dragging;
  float scale;
};

kernel void imagesComputeShader(
    device CellBase *cells [[buffer(0)]],
    device CellBase *outputLamps [[buffer(1)]],
    constant Params &params [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {

  // check out of bounds
  if (int(id) >= params.itemsCount) {
    return;
  }

  CellBase cell = cells[id];
  device CellBase &outputCell = outputLamps[id];

  float y = cell.position.y;
  if ((y < 0.5 || y > 2.5) && !cell.dragging) {
    // Slow clockwise rotation around camera position
    float3 cameraPos = params.viewerPosition;
    float3 vectorToCell = cell.position - cameraPos;

    // Project to horizontal plane for rotation
    float3 horizontalVector = float3(vectorToCell.x, 0.0, vectorToCell.z);
    float distance = length(horizontalVector);

    // Apply slow rotation (adjust rotationSpeed for desired speed)
    float angle = -params.time * 0.05 * pow(distance, 0.2);
    if (y < 0.5) {
      angle = -angle;
    }

    // Rotate the position (clockwise around y-axis)
    float cosAngle = cos(angle);
    float sinAngle = sin(angle);
    float3 rotated = float3(
        cosAngle * horizontalVector.x + sinAngle * horizontalVector.z,
        0.0,
        -sinAngle * horizontalVector.x + cosAngle * horizontalVector.z);

    // Maintain original height and normalize to original distance
    rotated = normalize(rotated) * distance;
    cell.position = cameraPos + float3(rotated.x, vectorToCell.y, rotated.z);
  }

  outputCell.position = cell.position;
  outputCell.color = cell.color;
  outputCell.lampIdf = cell.lampIdf;
}

vertex ImageInOut imagesVertexShader(
    ImageVertexIn in [[stage_in]],
    ushort amp_id [[amplification_id]],
    constant Uniforms &uniforms [[buffer(BufferIndexUniforms)]],
    constant TintUniforms &tintUniform [[buffer(BufferIndexTintUniforms)]],
    constant Params &params [[buffer(BufferIndexParams)]],
    const device CellBase *blocksData [[buffer(BufferIndexBase)]]) {
  ImageInOut out;

  UniformsPerView uniformsPerView = uniforms.perView[amp_id];
  float3 cameraAt = uniforms.cameraPos;
  CellBase cell = blocksData[in.seed];
  float3 basePosition = cell.position;

  float3 cameraDirection = basePosition - cameraAt;
  float3 upDirection = float3(0.0, 1.0, 0.0);
  float3 rightDirection = normalize(cross(cameraDirection, upDirection));

  float3 boxX = rightDirection * in.position.x * cell.scale;
  float3 boxY = upDirection * in.position.y * cell.scale;
  float4 position = float4(basePosition + boxX + boxY, 1.0);

  out.originalPosition = basePosition.xyz;
  out.uv = in.uv;

  position.w = 1.0; // reset w to 1.0 for the projection matrix

  out.position = uniformsPerView.modelViewProjectionMatrix * position;
  out.color = float4(in.color, tintUniform.tintOpacity);
  // Premultiply color channel by alpha channel.
  out.color.rgb = out.color.rgb * out.color.a;
  out.height = in.height;

  return out;
}

fragment float4 imagesFragmentShader(
    ImageInOut in [[stage_in]], texture2d<float> imageTexture [[texture(0)]]) {
  if (in.color.a <= 0.0) {
    discard_fragment();
  }

  // Check if texture is valid - will appear as solid red if texture is missing
  if (!is_null_texture(imageTexture)) {
    constexpr sampler textureSampler(mag_filter::linear, min_filter::linear);
    float4 textureColor = imageTexture.sample(textureSampler, in.uv);

    float y = in.originalPosition.y;
    bool nearCamera = y > 0.5 && y < 2.5;
    float alpha = 1;
    if (!nearCamera) {
      alpha = 0.2;
    }

    // Ensure all color channels are preserved correctly
    // Don't multiply colors - this was losing the blue channel
    return float4(textureColor.rgb * alpha, textureColor.a * in.color.a);
  } else {
    // Return red to indicate missing texture
    return float4(1.0, 0.0, 0.0, in.color.a);
  }
}
