/*
 See the LICENSE.txt file for this sample's licensing information.

 Abstract:
 Wind tunnel particle demo.
 Particles enter from the left inlet, move right through a horizontal
 cylindrical tunnel, interact with a sphere obstacle, and recycle at the outlet.

 Physics terms used:
 - Advection: base flow in +x
 - Inter-particle pressure: short-range repulsion/compression response
 - Inter-particle viscosity: short-range velocity smoothing
 - Solid collisions: tunnel wall + central sphere, with slight rebound
 */

// clang-format off
#include <metal_stdlib>
#include <simd/simd.h>
#import "ShaderTypes.h"
#import "PathProperties.h"
// clang-format on

using namespace metal;

typedef struct {
  float3 position [[attribute(0)]];
  int lineNumber [[attribute(1)]];
  int groupNumber [[attribute(2)]];
  int cellSide [[attribute(3)]];
} WindTunnelVertexIn;

typedef struct {
  float4 position [[position]];
  float4 color;
} WindTunnelInOut;

typedef struct {
  float time;
  float elapsed;
  int groupSize;
  float3 viewerPosition;
  float viewerScale;
  float viewerRotation;
  int totalLines;
} WindTunnelParams;

struct WindTunnelBase {
  float3 position;
  float3 color;
  float3 velocity;
  float3 extra; // x = age, y = inletRadial, z = stuckTimer
};

constant float3 kTunnelCenter = float3(0.0, 0.0, -2.0);
constant float kTunnelHalfY = 0.6f; // square cross-section half-extent
constant float kTunnelHalfZ = 0.6f;
constant float kTunnelHalfLength = 1.5f; // total length = 3m
constant float kInletHalfY = 0.48f;      // inlet half-extent (slightly smaller)
constant float kInletHalfZ = 0.48f;
constant float kBaseFlowSpeed = 0.75f;

// Cube obstacle (oriented box, rotates mostly around z-axis with slight y-axis)
constant float3 kCubeCenter =
    float3(0.0, -0.15, -2.0); // slightly below tunnel center
constant float3 kCubeHalf = float3(0.17f, 0.17f, 0.17f); // slightly larger, not full height
constant float kCubeRotSpeedZ = 0.6f; // radians per second around z-axis (primary)
constant float kCubeRotSpeedY = 0.12f; // radians per second around y-axis (secondary, slower)

constant float kPressureRadius = 0.35f;    // interaction range (was 0.18)
constant float kPressureStiffness = 18.0f; // repulsion strength (was 12)
constant float kViscosity = 4.5f;          // momentum exchange (was 2.8)
constant int kNeighborPairs = 16;          // more neighbor samples (was 10)
constant float kTargetDensity = 1.35f;
constant float kGasDiffusion = 0.9f;
constant float kThermalJitter = 0.015f;

constant float kBounceRestitution = 0.65f; // slightly inelastic for stability
constant float kSpeedDamping = 1.0f;
constant float kSpeedRegulation = 3.0f; // soft drag toward base speed (per sec)
constant float kMaxAge = 36.0f;

static uint mixBits(uint x) {
  x ^= x >> 16u;
  x *= 0x7feb352du;
  x ^= x >> 15u;
  x *= 0x846ca68bu;
  x ^= x >> 16u;
  return x;
}

static float rand01(uint seed) {
  return float(mixBits(seed) & 0x00FFFFFFu) / 16777215.0f;
}

static float4 applyGestureViewer(
    float4 p, float3 viewerPosition, float viewerScale, float viewerRotation) {
  float cosT = cos(viewerRotation);
  float sinT = sin(viewerRotation);
  float x = p.x * cosT - p.z * sinT;
  float z = p.x * sinT + p.z * cosT;
  p.x = x;
  p.z = z;
  p *= viewerScale;
  p = p - float4(viewerPosition, 0.0f);
  return p;
}

// Gold / blue color alternating every 0.4s based on current time.
static float3 windTimeColor(float time) {
  int batch = int(floor(time / 0.4f));
  return (batch & 1) == 0 ? float3(1.0f, 0.45f, 0.0f) // orange
                          : float3(0.0f, 0.7f, 1.0f); // cyan
}

static void windInitState(
    uint lineIdx,
    uint spawnNonce,
    float time,
    thread float3 &outPos,
    thread float3 &outVel,
    thread float3 &outColor,
    thread float &outRadial) {
  // Random rectangular cross-section position on the inlet plane
  float h1 =
      rand01(lineIdx * 747796405u + spawnNonce * 2891336453u + 277803737u);
  float h2 =
      rand01(lineIdx * 3266489917u + spawnNonce * 668265263u + 2246822519u);
  float dy = (h1 * 2.0f - 1.0f) * kInletHalfY;
  float dz = (h2 * 2.0f - 1.0f) * kInletHalfZ;
  outPos =
      float3(-kTunnelHalfLength, kTunnelCenter.y + dy, kTunnelCenter.z + dz);
  outVel = float3(kBaseFlowSpeed, 0.0f, 0.0f);

  float radial01 =
      clamp(max(abs(dy) / kInletHalfY, abs(dz) / kInletHalfZ), 0.0f, 1.0f);
  outColor = windTimeColor(time);
  outRadial = radial01;
}

kernel void windTunnelComputeShader(
    device WindTunnelBase *particles [[buffer(0)]],
    device WindTunnelBase *outputParticles [[buffer(1)]],
    constant WindTunnelParams &params [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
  WindTunnelBase cell = particles[id];
  device WindTunnelBase &out = outputParticles[id];

  uint stride = uint(params.groupSize + 1);
  bool leading = (id % stride == 0);

  if (leading) {
    float3 pos = cell.position;
    float3 vel = cell.velocity;
    float age = cell.extra.x;
    float inletRadial = cell.extra.y;
    float stuckT = cell.extra.z;

    uint lineIdx = id / stride;
    float dt = params.elapsed * 0.38f;
    age += dt;

    // Particle still waiting to be emitted (negative age = delay).
    if (age < 0.0f) {
      out = cell;
      out.extra.x = age;
      return;
    }

    // Base tunnel flow (+x) and weak centerline guidance to keep coherent
    // stream.
    float3 rel = pos - kTunnelCenter;
    float3 flowTarget =
        float3(kBaseFlowSpeed, -rel.y * 0.14f, -(rel.z) * 0.14f);

    float3 pressureForce = float3(0.0f);
    float3 viscForce = float3(0.0f);
    float3 centroidAcc = float3(0.0f);
    float density = 0.0f;
    float neighborCount = 0.0f;

    // Lightweight particle-particle interactions (pressure + viscosity).
    // Each particle samples multiple deterministic neighbor lines and applies
    // short-range forces when they are physically close.
    for (int k = 0; k < kNeighborPairs; ++k) {
      uint hopA = uint(17 + k * 31);
      uint hopB = uint(29 + k * 47);

      uint nIdxA = (lineIdx + hopA) % uint(params.totalLines);
      uint nIdxB = (lineIdx + uint(params.totalLines) -
                    (hopB % uint(params.totalLines))) %
                   uint(params.totalLines);

      uint nIdA = nIdxA * stride;
      uint nIdB = nIdxB * stride;

      WindTunnelBase nbA = particles[nIdA];
      WindTunnelBase nbB = particles[nIdB];

      float3 rijA = pos - nbA.position;
      float3 rijB = pos - nbB.position;
      float dA = length(rijA);
      float dB = length(rijB);

      if (dA > 1e-4f && dA < kPressureRadius) {
        float q = 1.0f - (dA / kPressureRadius);
        pressureForce += normalize(rijA) * (kPressureStiffness * q * q);
        viscForce += (nbA.velocity - vel) * (kViscosity * q);
        centroidAcc += nbA.position;
        density += q;
        neighborCount += 1.0f;
      }
      if (dB > 1e-4f && dB < kPressureRadius) {
        float q = 1.0f - (dB / kPressureRadius);
        pressureForce += normalize(rijB) * (kPressureStiffness * q * q);
        viscForce += (nbB.velocity - vel) * (kViscosity * q);
        centroidAcc += nbB.position;
        density += q;
        neighborCount += 1.0f;
      }
    }

    // Gas-like diffusion: in sparse zones, steer toward local centroid and add
    // tiny thermal jitter so particles actively fill low-density void regions.
    float sparse = clamp(
        (kTargetDensity - density) / max(kTargetDensity, 1e-4f), 0.0f, 1.0f);
    float3 voidFill = float3(0.0f);
    if (neighborCount > 0.5f) {
      float3 centroid = centroidAcc / neighborCount;
      voidFill += (centroid - pos) * (kGasDiffusion * sparse);
    }
    float jitterPhase = params.time * 3.7f + float(lineIdx) * 2.113f;
    float3 thermal = normalize(float3(
                         cos(jitterPhase),
                         sin(jitterPhase * 1.31f),
                         sin(jitterPhase * 0.73f + 1.0f))) *
                     (kThermalJitter * sparse);

    float3 accel = (flowTarget - vel) * 2.4f + pressureForce + viscForce +
                   voidFill + thermal;

    // Integrate velocity, then position.
    vel += accel * dt;
    vel *= kSpeedDamping;

    // Soft speed regulation toward base flow speed.
    float vlen = length(vel);
    if (vlen > 1e-4f) {
      float targetSpeed = kBaseFlowSpeed;
      float newSpeed =
          mix(vlen, targetSpeed, clamp(kSpeedRegulation * dt, 0.0f, 1.0f));
      vel = (vel / vlen) * newSpeed;
    } else {
      vel = float3(kBaseFlowSpeed * 0.5f, 0.0f, 0.0f);
    }

    float3 newPos = pos + vel * dt;

    // ── Rotating cube collision (post-integration) ──────────────
    // Transform newPos/vel into cube's local frame, do AABB test,
    // push out & reflect, then transform back.  Running AFTER
    // integration guarantees the final position is always outside.
    float cubeAngleZ = params.time * kCubeRotSpeedZ;
    float cosZ = cos(cubeAngleZ);
    float sinZ = sin(cubeAngleZ);
    float cubeAngleY = params.time * kCubeRotSpeedY;
    float cosY = cos(cubeAngleY);
    float sinY = sin(cubeAngleY);

    float3 relW = newPos - kCubeCenter;
    // world -> local: inverse Z rotation, then inverse Y rotation
    float3 relZInv = float3(
      relW.x * cosZ + relW.y * sinZ,
      -relW.x * sinZ + relW.y * cosZ,
      relW.z);
    float3 relLocal = float3(
      relZInv.x * cosY - relZInv.z * sinY,
      relZInv.y,
      relZInv.x * sinY + relZInv.z * cosY);

    float3 velZInv = float3(
      vel.x * cosZ + vel.y * sinZ,
      -vel.x * sinZ + vel.y * cosZ,
      vel.z);
    float3 velLocal = float3(
      velZInv.x * cosY - velZInv.z * sinY,
      velZInv.y,
      velZInv.x * sinY + velZInv.z * cosY);

    // Expand half-extents by a small skin so particles never tunnel through.
    const float kSkin = 0.005f;
    float3 cubeMin = -kCubeHalf;
    float3 cubeMax = kCubeHalf;

    bool inside =
        (relLocal.x >= cubeMin.x && relLocal.x <= cubeMax.x &&
         relLocal.y >= cubeMin.y && relLocal.y <= cubeMax.y &&
         relLocal.z >= cubeMin.z && relLocal.z <= cubeMax.z);

    if (inside) {
      // Push out along axis of least penetration
      float3 pen = float3(
          min(relLocal.x - cubeMin.x, cubeMax.x - relLocal.x),
          min(relLocal.y - cubeMin.y, cubeMax.y - relLocal.y),
          min(relLocal.z - cubeMin.z, cubeMax.z - relLocal.z));
      float3 faceN = float3(0.0f);
      if (pen.x <= pen.y && pen.x <= pen.z) {
        faceN.x = (relLocal.x < 0.0f) ? -1.0f : 1.0f;
        relLocal.x =
            (faceN.x < 0.0f) ? (cubeMin.x - kSkin) : (cubeMax.x + kSkin);
      } else if (pen.y <= pen.x && pen.y <= pen.z) {
        faceN.y = (relLocal.y < 0.0f) ? -1.0f : 1.0f;
        relLocal.y =
            (faceN.y < 0.0f) ? (cubeMin.y - kSkin) : (cubeMax.y + kSkin);
      } else {
        faceN.z = (relLocal.z < 0.0f) ? -1.0f : 1.0f;
        relLocal.z =
            (faceN.z < 0.0f) ? (cubeMin.z - kSkin) : (cubeMax.z + kSkin);
      }
      // Full elastic reflection (restitution = 1.0) for crisp bounce.
      float vn = dot(velLocal, faceN);
      if (vn < 0.0f) {
        velLocal -= 2.0f * vn * faceN;
      }
    }

    // Transform back to world frame: forward Y rotation, then forward Z rotation.
    float3 relYFwd = float3(
      relLocal.x * cosY + relLocal.z * sinY,
      relLocal.y,
      -relLocal.x * sinY + relLocal.z * cosY);
    float3 velYFwd = float3(
      velLocal.x * cosY + velLocal.z * sinY,
      velLocal.y,
      -velLocal.x * sinY + velLocal.z * cosY);

    newPos = kCubeCenter + float3(
                   relYFwd.x * cosZ - relYFwd.y * sinZ,
                   relYFwd.x * sinZ + relYFwd.y * cosZ,
                   relYFwd.z);
    vel = float3(
      velYFwd.x * cosZ - velYFwd.y * sinZ,
      velYFwd.x * sinZ + velYFwd.y * cosZ,
      velYFwd.z);

    // Tunnel wall collision (square cross-section) — bounce inward
    float dy = newPos.y - kTunnelCenter.y;
    float dz = newPos.z - kTunnelCenter.z;
    if (dy > kTunnelHalfY) {
      newPos.y = kTunnelCenter.y + kTunnelHalfY - 0.002f;
      if (vel.y > 0.0f)
        vel.y = -vel.y * kBounceRestitution;
    } else if (dy < -kTunnelHalfY) {
      newPos.y = kTunnelCenter.y - kTunnelHalfY + 0.002f;
      if (vel.y < 0.0f)
        vel.y = -vel.y * kBounceRestitution;
    }
    if (dz > kTunnelHalfZ) {
      newPos.z = kTunnelCenter.z + kTunnelHalfZ - 0.002f;
      if (vel.z > 0.0f)
        vel.z = -vel.z * kBounceRestitution;
    } else if (dz < -kTunnelHalfZ) {
      newPos.z = kTunnelCenter.z - kTunnelHalfZ + 0.002f;
      if (vel.z < 0.0f)
        vel.z = -vel.z * kBounceRestitution;
    }

    float spd = length(vel);
    stuckT = (spd < 0.05f) ? (stuckT + dt) : 0.0f;

    // Left wall bounce: reflect x-velocity to push particle rightward
    if (newPos.x < -kTunnelHalfLength) {
      newPos.x = -kTunnelHalfLength + 0.002f;
      if (vel.x < 0.0f) {
        vel.x = abs(vel.x);
      }
    }

    // Right wall: lifecycle ends → recycle from left at random inlet pos
    bool outRight = newPos.x > kTunnelHalfLength;
    bool tooOld = age > kMaxAge;
    bool stuck = stuckT > 1.2f;

    if (outRight || tooOld || stuck) {
      uint timeTick = uint(max(params.time, 0.0f) * 10000.0f);
      uint seed = mixBits(
          lineIdx * 2246822519u ^ timeTick * 3266489917u ^
          as_type<uint>(newPos.y * 997.0f) ^ as_type<uint>(newPos.z * 619.0f) ^
          as_type<uint>(age * 431.0f));
      float3 c;
      float radial;
      windInitState(lineIdx, seed, params.time, newPos, vel, c, radial);
      cell.color = c;
      inletRadial = radial;
      age = 0.0f;
      stuckT = 0.0f;
    }

    out.position = newPos;
    out.velocity = vel;
    out.color = cell.color;
    out.extra = float3(age, inletRadial, stuckT);

  } else {
    out = outputParticles[id - 1];
  }
}

vertex WindTunnelInOut windTunnelVertexShader(
    WindTunnelVertexIn in [[stage_in]],
    ushort amp_id [[amplification_id]],
    constant Uniforms &uniforms [[buffer(BufferIndexUniforms)]],
    constant TintUniforms &tintUniform [[buffer(BufferIndexTintUniforms)]],
    constant WindTunnelParams &params [[buffer(BufferIndexParams)]],
    const device WindTunnelBase *linesData [[buffer(BufferIndexBase)]]) {
  WindTunnelInOut out;

  UniformsPerView uniformsPerView = uniforms.perView[amp_id];
  float3 cameraDir = uniforms.cameraDirection;

  int stride = params.groupSize + 1;
  int lineNumber = in.lineNumber;
  int grp = in.groupNumber;
  int side = in.cellSide;

  WindTunnelBase cell = linesData[lineNumber * stride + grp + 1];
  WindTunnelBase prevCell = linesData[lineNumber * stride + grp];

  float3 dir = cell.position - prevCell.position;
  float segLen = length(dir);
  if (segLen < 1e-5f)
    dir = float3(1e-4f, 0.0f, 0.0f);

  float ageDiff = abs(cell.extra.x - prevCell.extra.x);
  float3 brush = (segLen < 0.00025f || segLen > 0.45f || ageDiff > 0.5f)
                     ? float3(0.0f)
                     : normalize(cross(dir, cameraDir)) * 0.0008f;

  float4 position;
  if (side == 0)
    position = float4(prevCell.position + brush, 1.0f);
  else if (side == 1)
    position = float4(prevCell.position - brush, 1.0f);
  else if (side == 2)
    position = float4(cell.position + brush, 1.0f);
  else
    position = float4(cell.position - brush, 1.0f);

  position = applyGestureViewer(
      position,
      params.viewerPosition,
      params.viewerScale,
      params.viewerRotation);
  position.w = 1.0f;

  out.position = uniformsPerView.modelViewProjectionMatrix * position;

  float3 col = cell.color;
  // Slight brightening in core stream for tunnel feel.
  col *= (1.0f + (1.0f - cell.extra.y) * 0.15f);
  out.color = float4(col, 1.0f);
  return out;
}

fragment float4 windTunnelFragmentShader(WindTunnelInOut in [[stage_in]]) {
  if (in.color.a <= 0.0f)
    discard_fragment();
  return in.color;
}
