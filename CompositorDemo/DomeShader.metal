
#include <metal_stdlib>
#include <simd/simd.h>

// Include header shared between this Metal shader code and Swift/C code.
// (This needs to be imported first)
#import "ShaderTypes.h"

// executing Metal API commands.
#import "PathProperties.h"

using namespace metal;

struct DomeVertexIn {
  float3 position [[attribute(VertexAttributePosition)]];
  float3 color [[attribute(VertexAttributeColor)]];
  int seed [[attribute(VertexAttributeSeed)]];
};

struct DomeInOut {
  float4 position [[position]];
  float4 color;
  float3 fragPos;
  bool is_facing_camera;
};

typedef struct {
  float3 viewerPosition;
  float time;
  float viewerScale;
  float viewerRotation;
} Params;

// Optimized memory layout: group frequently accessed data together
struct SpherePoint {
    float3 position;     // Point position on the dome (12 bytes)
    float angularSpeed;  // Angular speed (radians/second) (4 bytes) - total 16 bytes, aligned to 16-byte boundary
    float3 rotationAxis; // Rotation axis (line direction through sphere center) (12 bytes)
    float pointId;       // Point ID (4 bytes) - total 16 bytes
};

struct SphereVertex {
  float3 position;
  float3 color;
  int seed;
};

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

kernel void domeComputeShader(
    device SpherePoint *spherePoints [[buffer(0)]],
    device SpherePoint *outputSpherePoints [[buffer(1)]],
    constant Params &params [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
  // Prevent out-of-bounds thread access
  if (id >= 80) {
    return;
  }

  SpherePoint point = spherePoints[id];
  SpherePoint outputPoint;

  float dt = params.time;
  if (dt > 0.1) {
    dt = 0.016;
  }

  // Rodrigues' rotation formula
  float3 v = point.position;
  float3 k = point.rotationAxis;
  float theta = point.angularSpeed * dt;
  float cos_theta = cos(theta);
  float sin_theta = sin(theta);
  float3 rotated_v =
      v * cos_theta + cross(k, v) * sin_theta + k * dot(k, v) * (1 - cos_theta);

  outputPoint.position = rotated_v;
  outputPoint.rotationAxis = point.rotationAxis;
  outputPoint.angularSpeed = point.angularSpeed;
  outputPoint.pointId = point.pointId;

  outputSpherePoints[id] = outputPoint;
}


vertex DomeInOut domeVertexShader(
    DomeVertexIn in [[stage_in]],
    ushort amp_id [[amplification_id]],
    constant Uniforms &uniforms [[buffer(BufferIndexUniforms)]],
    constant TintUniforms &tintUniform [[buffer(BufferIndexTintUniforms)]],
    constant Params &params [[buffer(BufferIndexParams)]],
    const device SpherePoint *spherePoints [[buffer(BufferIndexBase)]]) {
  DomeInOut out;

  UniformsPerView uniformsPerView = uniforms.perView[amp_id];

  // Object-space position and normal
  float4 position_obj = float4(in.position, 1.0);
  float3 normal_obj = normalize(-in.position);

  // Apply model transform to position to get world position
  float4 position_world = applyGestureViewerOnScene(
      position_obj,
      params.viewerPosition,
      params.viewerScale,
      params.viewerRotation,
      uniforms.cameraPos);

  // To get the world-space normal, we need to apply the rotation part of the transform.
  float cosTheta = cos(params.viewerRotation);
  float sinTheta = sin(params.viewerRotation);
  float normal_x = normal_obj.x * cosTheta - normal_obj.z * sinTheta;
  float normal_z = normal_obj.x * sinTheta + normal_obj.z * cosTheta;
  float3 normal_world = normalize(float3(normal_x, normal_obj.y, normal_z));

  // Calculate visibility
  float3 view_dir = normalize(uniforms.cameraPos - position_world.xyz);
  out.is_facing_camera = dot(normal_world, view_dir) > 0.0;

  // Optimization: pre-calculate if there are nearby points, set alpha to 0 if none
  float3 fragPos = in.position;
  float minDist = 10.0;  // Initialize to a large value
  
  // Quick check for nearby points
  for (uint i = 0; i < 80; i++) {
    float dist = distance(fragPos, spherePoints[i].position);
    minDist = min(minDist, dist);
    if (minDist < 1.6) break;  // Early exit
  }
  
  // If no nearby points, set alpha to 0 so fragment shader can discard
  float opacity = (minDist < 1.6) ? tintUniform.tintOpacity : 0.0;

  // Final position
  position_world.w = 1;
  out.position = uniformsPerView.modelViewProjectionMatrix * position_world;

  // Pass object-space position to fragment shader via 'color'
  out.color = float4(in.position, opacity);

  return out;
}

// Optimized fragment shader - using spatial partitioning and early exit
fragment float4 domeFragmentShader(
    DomeInOut in [[stage_in]],
    constant Params &params [[buffer(BufferIndexParams)]],
    const device SpherePoint *spherePoints [[buffer(BufferIndexBase)]]) {

  if (!in.is_facing_camera) {
    return float4(0.2, 0.2, 0.2, 1.0);
  }

  if (in.color.a <= 0.0) {
    discard_fragment();
  }

  // Current fragment position on the dome
  float3 fragPos = in.color.xyz;
  float3 frag_norm = normalize(fragPos);

  // Default transparent
  float4 finalColor = float4(0.0, 0.0, 0.0, 0.0);
  
  // Optimization 1: pre-calculate common values
  const float pointRadius = 0.05;
  const float maxConnectionDist = 1.5;
  const float lineThickness = 0.002;
  const float searchRadius = 1.52;
  
  // Optimization 2: use smaller search radius and early exit
  uint nearbyPoints[16]; // Store at most 16 nearby points
  uint nearbyCount = 0;
  
  // First pass: find nearby points
  for (uint i = 0; i < 80 && nearbyCount < 16; i++) {
    float3 point1 = spherePoints[i].position;
    float dist = distance(fragPos, point1);
    
    // Optimization 3: early exit for point rendering
    if (dist < pointRadius) {
      return float4(1.0, 1.0, 1.0, 1.0);
    }
    
    // Only consider points within search radius
    if (dist < searchRadius) {
      nearbyPoints[nearbyCount] = i;
      nearbyCount++;
    }
  }
  
  // Optimization 4: only check connections between nearby points
  for (uint i = 0; i < nearbyCount; i++) {
    uint idx1 = nearbyPoints[i];
    float3 point1 = spherePoints[idx1].position;
    float3 p1_norm = normalize(point1);
    
    for (uint j = i + 1; j < nearbyCount; j++) {
      uint idx2 = nearbyPoints[j];
      float3 point2 = spherePoints[idx2].position;
      float lineLength = distance(point1, point2);
      
      // Optimization 5: stricter distance check
      if (lineLength > 1e-5 && lineLength < maxConnectionDist) {
        float3 p2_norm = normalize(point2);
        
        // Optimization 6: pre-calculate dot product
        float p1_p2_dot = dot(p1_norm, p2_norm);
        float frag_p1_dot = dot(frag_norm, p1_norm);
        float frag_p2_dot = dot(frag_norm, p2_norm);
        
        // Early exit: skip if not on arc
        if (frag_p1_dot < p1_p2_dot || frag_p2_dot < p1_p2_dot) {
          continue;
        }
        
        // Calculate distance to great circle
        float3 n = normalize(cross(p1_norm, p2_norm));
        float dist_from_plane = abs(dot(frag_norm, n));
        
        if (dist_from_plane < lineThickness) {
          return float4(1.0, 1.0, 1.0, 1.0);
        }
      }
    }
  }

  return finalColor;
}
