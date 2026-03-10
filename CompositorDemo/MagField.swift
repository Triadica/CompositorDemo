/*
 See the LICENSE.txt file for this sample's licensing information.

 Abstract:
 Earth magnetic-field line demo.
 Two groups of charged particles (positive = orange-red, negative = cyan-blue)
 are emitted from x = -5 m and travel through Earth's magnetic dipole field.
 The Lorentz force  F = q(v × B)  curves them in opposite directions,
 tracing out the characteristic figure-of-eight field-line topology.
 */

import CompositorServices
import Metal
import MetalKit
import Spatial
import SwiftUI
import simd

// ─── Constants ────────────────────────────────────────────────────────────────

private let maxFramesInFlight = 3

/// Total number of independent particle trails
private let linesCount: Int = 80_000
/// Number of trail segments per particle (history length)
private let lineGroupSize: Int = 4
/// Entries per trail in the compute buffer (1 leader + N trail)
private var controlCountPerLine: Int { lineGroupSize + 1 }
private var controlCount: Int { linesCount * controlCountPerLine }

private let verticesCount: Int = controlCount * 6
private let indexesCount: Int = controlCount * 6

// ─── CPU-side structs (must match MagField.metal byte-for-byte) ───────────────

/// Mirrors Metal's `hashFloat()` — must stay in sync with MagField.metal.
private func magHashFloat(_ seed: UInt32) -> Float {
  var s = seed
  s ^= s << 13
  s ^= s >> 17
  s ^= s << 5
  return Float(s & 0x000F_FFFF) / Float(0x000F_FFFF)
}

private func magMixBits(_ x: UInt32) -> UInt32 {
  var v = x
  v ^= v >> 16
  v &*= 0x7feb_352d
  v ^= v >> 15
  v &*= 0x846c_a68b
  v ^= v >> 16
  return v
}

private func magRand01(_ seed: UInt32) -> Float {
  Float(magMixBits(seed) & 0x00FF_FFFF) / 16_777_215.0
}

/// Matches `struct MagFieldBase` in MagField.metal.
/// Four SIMD3<Float> → 4 × 12 bytes = 48 bytes (Swift alignment matches Metal).
private struct MagFieldBase {
  var position: SIMD3<Float>
  var color: SIMD3<Float>
  var velocity: SIMD3<Float>
  var extra: SIMD3<Float>  // x = charge (+1 or −1)
}

/// Matches `typedef struct MagFieldParams` in MagField.metal.
/// In Metal, float3 has size=12 and alignment=16, so the struct layout is:
///   time(4) elapsed(4) groupSize(4) [4-pad] viewerPosition(12)
///   viewerScale(4@28) viewerRotation(4@32) totalLines(4@36) [8-pad]
/// Swift SIMD3<Float> likewise has size=12 and alignment=16.
/// The layouts match — only use .stride (48) not .size (40) when passing to Metal.
private struct Params {
  var time: Float
  var elapsed: Float
  var groupSize: Int32 = Int32(lineGroupSize)
  var viewerPosition: SIMD3<Float>  // offset 16 (after 4-byte auto-pad)
  var viewerScale: Float  // offset 28
  var viewerRotation: Float = 0.0  // offset 32
  var totalLines: Int32 = Int32(linesCount)  // offset 36
}

// ─── Renderer ─────────────────────────────────────────────────────────────────

@MainActor
class MagFieldRenderer: CustomRenderer {
  private let renderPipelineState: MTLRenderPipelineState & Sendable

  private var uniformsBuffer: [MTLBuffer]
  var vertexBuffer: MTLBuffer!
  var indexBuffer: MTLBuffer!

  let computeDevice: MTLDevice
  var computeBuffer: PingPongBuffer?
  let computePipeLine: MTLComputePipelineState
  let computeCommandQueue: MTLCommandQueue

  var gestureManager = GestureManager()

  // Track frame time
  private var viewStartTime: Date = Date()
  private var frameDelta: Float = 0.0

  // MARK: – Init

  init(layerRenderer: LayerRenderer) throws {
    uniformsBuffer = (0..<Renderer.maxFramesInFlight).map { _ in
      layerRenderer.device.makeBuffer(length: MemoryLayout<PathProperties>.uniformStride)!
    }

    renderPipelineState = try Self.makeRenderPipelineDescriptor(layerRenderer: layerRenderer)

    computeDevice = MTLCreateSystemDefaultDevice()!
    let library = computeDevice.makeDefaultLibrary()!
    let kernelFn = library.makeFunction(name: "magFieldComputeShader")!
    computePipeLine = try computeDevice.makeComputePipelineState(function: kernelFn)
    computeCommandQueue = computeDevice.makeCommandQueue()!

    createVertexBuffer(device: layerRenderer.device)
    createIndexBuffer(device: layerRenderer.device)
    createComputeBuffer(device: layerRenderer.device)
  }

  // MARK: – Buffer creation

  private func createVertexBuffer(device: MTLDevice) {
    let len = MemoryLayout<AttractorCellVertex>.stride * verticesCount
    vertexBuffer = device.makeBuffer(length: len)!
    vertexBuffer.label = "MagField vertex buffer"
    let verts = vertexBuffer.contents().assumingMemoryBound(to: AttractorCellVertex.self)

    for i in 0..<linesCount {
      let base = i * lineGroupSize * 6
      for j in 0..<lineGroupSize {
        let idx = base + j * 6
        let sides: [Int32] = [0, 1, 2, 1, 2, 3]
        for (k, side) in sides.enumerated() {
          verts[idx + k] = AttractorCellVertex(
            position: SIMD3<Float>(0, 0, 0),
            lineNumber: Int32(i),
            groupNumber: Int32(j),
            cellSide: side)
        }
      }
    }
  }

  private func createIndexBuffer(device: MTLDevice) {
    let len = MemoryLayout<UInt32>.stride * indexesCount
    indexBuffer = device.makeBuffer(length: len)!
    indexBuffer.label = "MagField index buffer"
    let idxPtr = indexBuffer.contents().bindMemory(to: UInt32.self, capacity: indexesCount)
    for i in 0..<indexesCount { idxPtr[i] = UInt32(i) }
  }

  func resetComputeState() {
    createComputeBuffer(device: computeDevice)
  }

  private func createComputeBuffer(device: MTLDevice) {
    let len = MemoryLayout<MagFieldBase>.stride * controlCount
    computeBuffer = PingPongBuffer(device: device, length: len)
    guard let buf = computeBuffer else { return }
    buf.addLabel("MagField compute buffer")

    let ptr = buf.currentBuffer.contents().bindMemory(to: MagFieldBase.self, capacity: controlCount)

    // 2×2×2 box centred at (0, 0, -2), half-extent = 1
    let boxCenterY: Float = 0.0
    let boxCenterZ: Float = -2.0
    let boxHalf: Float = 1.0
    let flowSpeed: Float = 0.3
    let spawnWindow: Float = 3.0
    let half = linesCount / 2

    for i in 0..<linesCount {
      let positive = i < half
      let color: SIMD3<Float> =
        positive
        ? SIMD3<Float>(1.0, 0.38, 0.07)  // orange-red  (+)
        : SIMD3<Float>(0.07, 0.48, 1.0)  // cyan-blue   (-)
      let charge: Float = positive ? 1.0 : -1.0

      // Random and uniform YZ on the left face (decorrelated seeds)
      let h1 = magRand01(UInt32(i) &* 747_796_405 &+ 277_803_737)
      let h2 = magRand01(UInt32(i) &* 3_266_489_917 &+ 2_246_822_519)
      let hAge = magRand01(UInt32(i) &* 668_265_263 &+ 0xa511_e9b3)
      let y = boxCenterY + (h1 * 2.0 - 1.0) * boxHalf
      let z = boxCenterZ + (h2 * 2.0 - 1.0) * boxHalf

      // Initial state also emits from left face
      let px = -boxHalf  // boxCenter.x = 0
      let emitPos = SIMD3<Float>(px, y, z)

      let vel = SIMD3<Float>(flowSpeed, 0.0, 0.0)
      let initialAge = -hAge * spawnWindow

      for j in 0..<controlCountPerLine {
        let index = i * controlCountPerLine + j
        ptr[index] = MagFieldBase(
          position: emitPos,
          color: color,
          velocity: vel,
          extra: SIMD3<Float>(charge, initialAge, 0))
      }
    }

    buf.copyToNext()
  }

  // MARK: – Pipeline

  class func buildMetalVertexDescriptor() -> MTLVertexDescriptor {
    let desc = MTLVertexDescriptor()
    var offset = 0

    desc.attributes[0].format = .float3
    desc.attributes[0].offset = offset
    desc.attributes[0].bufferIndex = 0
    offset += MemoryLayout<SIMD3<Float>>.stride

    desc.attributes[1].format = .int
    desc.attributes[1].offset = offset
    desc.attributes[1].bufferIndex = 0
    offset += MemoryLayout<Int32>.stride

    desc.attributes[2].format = .int
    desc.attributes[2].offset = offset
    desc.attributes[2].bufferIndex = 0
    offset += MemoryLayout<Int32>.stride

    desc.attributes[3].format = .int
    desc.attributes[3].offset = offset
    desc.attributes[3].bufferIndex = 0
    offset += MemoryLayout<Int32>.stride

    desc.layouts[0].stride = MemoryLayout<AttractorCellVertex>.stride
    desc.layouts[0].stepRate = 1
    desc.layouts[0].stepFunction = .perVertex
    return desc
  }

  private static func makeRenderPipelineDescriptor(layerRenderer: LayerRenderer) throws
    -> MTLRenderPipelineState
  {
    let pipeDesc = Renderer.defaultRenderPipelineDescriptor(layerRenderer: layerRenderer)
    let library = layerRenderer.device.makeDefaultLibrary()!
    pipeDesc.vertexFunction = library.makeFunction(name: "magFieldVertexShader")
    pipeDesc.fragmentFunction = library.makeFunction(name: "magFieldFragmentShader")
    pipeDesc.label = "MagFieldRenderPipeline"
    pipeDesc.vertexDescriptor = buildMetalVertexDescriptor()
    return try layerRenderer.device.makeRenderPipelineState(descriptor: pipeDesc)
  }

  // MARK: – Draw command

  func drawCommand(frame: LayerRenderer.Frame) throws -> TintDrawCommand {
    return TintDrawCommand(
      frameIndex: frame.frameIndex,
      uniforms: uniformsBuffer[Int(frame.frameIndex % Renderer.maxFramesInFlight)],
      verticesCount: verticesCount)
  }

  // MARK: – Compute

  func computeCommandCommit() {
    guard let computeBuffer,
      let cmdBuf = computeCommandQueue.makeCommandBuffer(),
      let encoder = cmdBuf.makeComputeCommandEncoder()
    else { return }

    let delta = -Float(viewStartTime.timeIntervalSinceNow)
    let dt = delta - frameDelta
    frameDelta = delta

    var params = Params(
      time: delta,
      elapsed: dt,
      viewerPosition: gestureManager.viewerPosition,
      viewerScale: gestureManager.viewerScale,
      viewerRotation: gestureManager.viewerRotation)

    encoder.setComputePipelineState(computePipeLine)
    encoder.setBuffer(computeBuffer.currentBuffer, offset: 0, index: 0)
    encoder.setBuffer(computeBuffer.nextBuffer, offset: 0, index: 1)
    encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 2)

    let tgs = min(computePipeLine.maxTotalThreadsPerThreadgroup, 256)
    let threadsPerGroup = MTLSize(width: tgs, height: 1, depth: 1)
    let groups = MTLSize(width: (controlCount + tgs - 1) / tgs, height: 1, depth: 1)
    encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threadsPerGroup)
    encoder.endEncoding()
    cmdBuf.commit()
    computeBuffer.swap()
  }

  // MARK: – Render encode

  func encodeDraw(
    _ drawCommand: TintDrawCommand,
    encoder: MTLRenderCommandEncoder,
    drawable: LayerRenderer.Drawable,
    device: MTLDevice,
    tintValue: Float,
    buffer: MTLBuffer,
    indexBuffer: MTLBuffer
  ) {
    encoder.setCullMode(.none)
    encoder.setRenderPipelineState(renderPipelineState)

    var demoUniform = TintUniforms(tintOpacity: tintValue)
    encoder.setVertexBytes(
      &demoUniform,
      length: MemoryLayout<TintUniforms>.size,
      index: BufferIndex.tintUniforms.rawValue)
    encoder.setVertexBuffer(
      drawCommand.uniforms, offset: 0,
      index: BufferIndex.uniforms.rawValue)
    encoder.setVertexBuffer(
      buffer, offset: 0,
      index: BufferIndex.meshPositions.rawValue)

    let delta = -Float(viewStartTime.timeIntervalSinceNow)
    let dt = delta - frameDelta

    var params = Params(
      time: delta,
      elapsed: dt,
      viewerPosition: gestureManager.viewerPosition,
      viewerScale: gestureManager.viewerScale,
      viewerRotation: gestureManager.viewerRotation)

    let paramsBuf = device.makeBuffer(
      bytes: &params,
      length: MemoryLayout<Params>.stride,
      options: .storageModeShared)!
    encoder.setVertexBuffer(paramsBuf, offset: 0, index: BufferIndex.params.rawValue)
    encoder.setVertexBuffer(
      computeBuffer?.currentBuffer, offset: 0,
      index: BufferIndex.base.rawValue)

    encoder.drawIndexedPrimitives(
      type: .triangle,
      indexCount: indexesCount,
      indexType: .uint32,
      indexBuffer: indexBuffer,
      indexBufferOffset: 0)
  }

  func updateUniformBuffers(_ drawCommand: TintDrawCommand, drawable: LayerRenderer.Drawable) {
    drawCommand.uniforms.contents()
      .assumingMemoryBound(to: Uniforms.self).pointee = Uniforms(drawable: drawable)
  }

  func onSpatialEvents(events: SpatialEventCollection) {
    for event in events { gestureManager.onSpatialEvent(event: event) }
  }
}
