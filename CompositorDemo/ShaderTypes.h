/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
Header containing types and enum constants shared between Metal shaders and
Swift/ObjC source.
*/

#ifndef ShaderTypes_h
#define ShaderTypes_h

#include <simd/vector_types.h>
#ifdef __METAL_VERSION__
#define NS_ENUM(_type, _name)                                                  \
  enum _name : _type _name;                                                    \
  enum _name : _type
typedef metal::int32_t EnumBackingType;
#else
#import <Foundation/Foundation.h>
typedef NSInteger EnumBackingType;
#endif

#include <simd/simd.h>

typedef NS_ENUM(EnumBackingType, BufferIndex) {
  BufferIndexMeshPositions = 0,
  BufferIndexUniforms = 1,
  BufferIndexTintUniforms = 2,
  BufferIndexParams = 3,
  BufferIndexBase = 4,
  BufferIndexCount
};

typedef NS_ENUM(EnumBackingType, VertexAttribute) {
  VertexAttributePosition = 0,
  VertexAttributeColor = 1,
  VertexAttributeSeed = 2,
};

typedef NS_ENUM(EnumBackingType, PolylineVertexAttribute) {
  PolylineVertexAttributePosition = 0,
  PolylineVertexAttributeColor = 1,
  PolylineVertexAttributeDirection = 2,
  PolylineVertexAttributeSeed = 3,
};

typedef struct {
  matrix_float4x4 modelViewProjectionMatrix;
} UniformsPerView;

typedef struct {
  UniformsPerView perView[2];
  simd_float3 cameraPos;
  simd_float3 cameraDirection;
} Uniforms;

typedef struct {
  float tintOpacity;
} TintUniforms;

typedef struct {
  simd_float3 position;
  simd_float3 color;
  int seed;
} VertexWithSeed;

typedef struct {
  simd_float3 position;
  simd_float3 color;
  simd_float3 direction;
  int seed;
} PolylineVertex;

typedef struct {
  simd_float3 position;
  int lineNumber;
  int groupNumber;
  int cellSide;
} AttractorCellVertex;

typedef struct {
  simd_float3 position;
  simd_float3 color;
  int seed;
  float height;
  simd_float2 uv;
} BlockVertex;

typedef struct {
  matrix_float4x4 modelMatrix;
  matrix_float3x3 normalMatrix;
} RenderParams;

#endif /* ShaderTypes_h */
