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
} AttractorVertexIn;

typedef struct {
  float4 position [[position]];
  float4 color;

} AttractorInOut;

typedef struct {
  float time;
} AttractorParams;

struct AttractorBase {
  float3 position;
  float3 color;
  float lampIdf;
  float3 velocity;
};

static float random1D(float seed) { return fract(sin(seed) * 43758.5453123); }

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

static int getGroupSize() { return 20; }

kernel void attractorComputeShader(
    device AttractorBase *attractor [[buffer(0)]],
    device AttractorBase *outputAttractor [[buffer(1)]],
    constant AttractorParams &params [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
  AttractorBase lamp = attractor[id];
  device AttractorBase &outputCell = outputAttractor[id];

  bool leading = (id % (getGroupSize() + 1) == 0);

  if (leading) {
    // float seed = fract(lamp.lampIdf / 10.) * 10.;
    // float speed = random1D(seed) + 0.1;
    float dt = params.time * 0.1; // TODO maybe remove this
    outputCell.position = fourwingLineIteration(lamp.position, dt);
    outputCell.color = lamp.color;
    outputCell.lampIdf = lamp.lampIdf;
  } else {
    // copy previous
    outputCell.position = outputAttractor[id - 1].position;
    outputCell.color = outputAttractor[id - 1].position;
    outputCell.lampIdf = outputAttractor[id - 1].lampIdf;
  }
}

vertex AttractorInOut attractorVertexShader(
    AttractorVertexIn in [[stage_in]],
    ushort amp_id [[amplification_id]],
    constant Uniforms &uniforms [[buffer(BufferIndexUniforms)]],
    constant TintUniforms &tintUniform [[buffer(BufferIndexTintUniforms)]],
    constant AttractorParams &params [[buffer(BufferIndexParams)]],
    const device AttractorBase *linesData [[buffer(BufferIndexBase)]]) {
  AttractorInOut out;

  UniformsPerView uniformsPerView = uniforms.perView[amp_id];
  // float3 cameraAt = uniforms.cameraPos;
  simd_float3 cameraDirection = uniforms.cameraDirection;

  int lineNumber = in.lineNumber;
  int groupNumber = in.groupNumber;
  int cellSide = in.cellSide;

  AttractorBase cell =
      linesData[lineNumber * (getGroupSize() + 1) + groupNumber + 1];
  AttractorBase prevCell =
      linesData[lineNumber * (getGroupSize() + 1) + groupNumber];

  // float direction = cell.position - prevCell.position;
  // float3 brush = normalize(cross(direction, cameraDirection)) * 0.0001;

  float3 perpWidth = float3(0.0, 1.0, 0.0);

  float4 position = float4(0., 0., 0., 1.0);
  if (cellSide == 0) {
    position = float4(prevCell.position + perpWidth * 0.1, 1.0);
  } else if (cellSide == 1) {
    position = float4(prevCell.position - perpWidth * 0.1, 1.0);
  } else if (cellSide == 2) {
    position = float4(cell.position + perpWidth * 0.1, 1.0);
  } else if (cellSide == 3) {
    position = float4(cell.position - perpWidth * 0.1, 1.0);
  }

  out.position = uniformsPerView.modelViewProjectionMatrix *
                 (position * 0.2 + float4(0.0, 0.0, -2.0, 0.));
  out.color = float4(cell.color, tintUniform.tintOpacity);

  return out;
}

fragment float4 attractorFragmentShader(AttractorInOut in [[stage_in]]) {
  if (in.color.a <= 0.0) {
    discard_fragment();
  }

  // return in.color;
  return float4(1.0, 1.0, 1.0, 1.0);
}
