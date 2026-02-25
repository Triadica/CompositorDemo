/*
 See the LICENSE.txt file for this sample's licensing information.

 Abstract:
 Schwarzschild black-hole photon-trajectory demo.
 50 000 photon trails orbit (or get captured by / escape from) a black hole
 centred at (0, 0, −2).  Physics: null geodesics via Runge-Kutta 4.
 */

import CompositorServices
import Metal
import MetalKit
import Spatial
import SwiftUI
import simd

// ─── Constants ────────────────────────────────────────────────────────────────

private let maxFramesInFlight = 3

/// Number of independent photon trails.
private let linesCount: Int = 50_000
/// History segments per trail (doubled again: 4 → 8).
private let lineGroupSize: Int = 8
private var controlCountPerLine: Int { lineGroupSize + 1 }
private var controlCount: Int { linesCount * controlCountPerLine }
private let verticesCount: Int = controlCount * 6
private let indexesCount: Int = controlCount * 6

/// Black hole centre in world space (matches kBHCenter in BlackHole.metal).
private let bhCenter: SIMD3<Float> = SIMD3<Float>(0, 0, -2)
/// Emission sphere radius (matches kEmitR in metal shader).
private let emitRadius: Float = 20.0

// ─── CPU structs — must match BlackHole.metal byte-for-byte ──────────────────

/// Matches `struct BHBase`.  Four SIMD3<Float> = 4 × 12 = 48 bytes
/// (SIMD3<Float>.alignment = 4, so no inter-field padding).
private struct BHBase {
  var position: SIMD3<Float>
  var color: SIMD3<Float>
  var velocity: SIMD3<Float>
  var extra: SIMD3<Float>  // x = age, y = colorParam, z = stuckTimer
}

/// Matches `typedef struct BHParams`.
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
  var totalLines: Int32  // offset 36
}

// ─── Renderer ─────────────────────────────────────────────────────────────────

@MainActor
class BlackHoleRenderer: CustomRenderer {
  private let renderPipelineState: MTLRenderPipelineState & Sendable

  private var uniformsBuffer: [MTLBuffer]
  var vertexBuffer: MTLBuffer!
  var indexBuffer: MTLBuffer!

  let computeDevice: MTLDevice
  var computeBuffer: PingPongBuffer?
  let computePipeLine: MTLComputePipelineState
  let computeCommandQueue: MTLCommandQueue

  var gestureManager = GestureManager()
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
    let kernelFn = library.makeFunction(name: "blackHoleComputeShader")!
    computePipeLine = try computeDevice.makeComputePipelineState(function: kernelFn)
    computeCommandQueue = computeDevice.makeCommandQueue()!

    buildVertexBuffer(device: layerRenderer.device)
    buildIndexBuffer(device: layerRenderer.device)
    buildComputeBuffer(device: layerRenderer.device)
  }

  // MARK: – Buffer creation

  private func buildVertexBuffer(device: MTLDevice) {
    let len = MemoryLayout<AttractorCellVertex>.stride * verticesCount
    vertexBuffer = device.makeBuffer(length: len)!
    vertexBuffer.label = "BlackHole vertex buffer"
    let v = vertexBuffer.contents().assumingMemoryBound(to: AttractorCellVertex.self)

    for i in 0..<linesCount {
      let base = i * lineGroupSize * 6
      for j in 0..<lineGroupSize {
        let idx = base + j * 6
        let sides: [Int32] = [0, 1, 2, 1, 2, 3]
        for (k, side) in sides.enumerated() {
          v[idx + k] = AttractorCellVertex(
            position: SIMD3<Float>(0, 0, 0),
            lineNumber: Int32(i),
            groupNumber: Int32(j),
            cellSide: side)
        }
      }
    }
  }

  private func buildIndexBuffer(device: MTLDevice) {
    let len = MemoryLayout<UInt32>.stride * indexesCount
    indexBuffer = device.makeBuffer(length: len)!
    indexBuffer.label = "BlackHole index buffer"
    let p = indexBuffer.contents().bindMemory(to: UInt32.self, capacity: indexesCount)
    for i in 0..<indexesCount { p[i] = UInt32(i) }
  }

  func resetComputeState() {
    buildComputeBuffer(device: computeDevice)
  }

  private func buildComputeBuffer(device: MTLDevice) {
    let len = MemoryLayout<BHBase>.stride * controlCount
    computeBuffer = PingPongBuffer(device: device, length: len)
    guard let buf = computeBuffer else { return }
    buf.addLabel("BlackHole compute buffer")

    let ptr = buf.currentBuffer.contents().bindMemory(to: BHBase.self, capacity: controlCount)

    for i in 0..<linesCount {
      // ── Emission position on sphere around BH center ───────────────────
      let fib = fibonacciGrid(n: Float(i), total: Float(linesCount))
      let emitPos = bhCenter + fib * emitRadius

      // ── Initial velocity ─────────────────────────────────────────────
      // Real Schwarzschild capture cross-section: photons with impact
      // parameter b < 3√3·M ≈ 1.56 are captured.  At r=20 that means only
      // a ~4.5° cone around the radial direction escapes.
      // We mimic this by using a very small tangential spread for most
      // photons.  Equatorial emitters (accretion disk, |y| < 0.28) get a
      // larger tangential component so some orbit and escape outward,
      // matching what is observed in real accretion-disk imagery.
      let fib2 = fibonacciGrid(n: Float(i + linesCount), total: Float(linesCount * 2))
      let absy = abs(fib.y)
      let diskFactor: Float = absy < 0.28 ? (1.0 - absy / 0.28) : 0.0  // 1 at equator
      let tangential: Float = 0.04 + diskFactor * 0.51  // 0.04 (polar) … 0.55 (equatorial)
      var velDir = fib2 * tangential + (-fib) * (1.0 - tangential)
      let vlen = simd_length(velDir)
      if vlen > 0.0001 { velDir /= vlen }

      // ── Color: latitude-driven warm ↔ cool gradient ────────────────────
      let latitude: Float = fib.y * 0.5 + 0.5  // 0..1, south→north
      let t: Float = Float(i % 1000) / 1000 * 0.6 + latitude * 0.4
      let warm = SIMD3<Float>(1.0, 0.72, 0.25)
      let cool = SIMD3<Float>(0.35, 0.75, 1.0)
      let color = warm * (1 - t) + cool * t

      for j in 0..<controlCountPerLine {
        let index = i * controlCountPerLine + j
        ptr[index] = BHBase(
          position: emitPos,
          color: color,
          velocity: velDir,
          extra: SIMD3<Float>(0, latitude, 0))  // x=age, y=colorParam, z=stuckT
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
    let pd = Renderer.defaultRenderPipelineDescriptor(layerRenderer: layerRenderer)
    let lib = layerRenderer.device.makeDefaultLibrary()!
    pd.vertexFunction = lib.makeFunction(name: "blackHoleVertexShader")
    pd.fragmentFunction = lib.makeFunction(name: "blackHoleFragmentShader")
    pd.label = "BlackHoleRenderPipeline"
    pd.vertexDescriptor = buildMetalVertexDescriptor()
    return try layerRenderer.device.makeRenderPipelineState(descriptor: pd)
  }

  // MARK: – Draw command

  func drawCommand(frame: LayerRenderer.Frame) throws -> TintDrawCommand {
    TintDrawCommand(
      frameIndex: frame.frameIndex,
      uniforms: uniformsBuffer[Int(frame.frameIndex % Renderer.maxFramesInFlight)],
      verticesCount: verticesCount)
  }

  // MARK: – Compute dispatch

  func computeCommandCommit() {
    guard let computeBuffer,
      let cmd = computeCommandQueue.makeCommandBuffer(),
      let enc = cmd.makeComputeCommandEncoder()
    else { return }

    let elapsed = -Float(viewStartTime.timeIntervalSinceNow)
    let dt = elapsed - frameDelta
    frameDelta = elapsed

    var params = Params(
      time: elapsed,
      elapsed: dt,
      viewerPosition: gestureManager.viewerPosition,
      viewerScale: gestureManager.viewerScale,
      viewerRotation: gestureManager.viewerRotation,
      totalLines: Int32(linesCount))

    enc.setComputePipelineState(computePipeLine)
    enc.setBuffer(computeBuffer.currentBuffer, offset: 0, index: 0)
    enc.setBuffer(computeBuffer.nextBuffer, offset: 0, index: 1)
    enc.setBytes(&params, length: MemoryLayout<Params>.stride, index: 2)

    let tgs = min(computePipeLine.maxTotalThreadsPerThreadgroup, 256)
    let tpg = MTLSize(width: tgs, height: 1, depth: 1)
    let groups = MTLSize(width: (controlCount + tgs - 1) / tgs, height: 1, depth: 1)
    enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tpg)
    enc.endEncoding()
    cmd.commit()
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

    var uni = TintUniforms(tintOpacity: tintValue)
    encoder.setVertexBytes(
      &uni, length: MemoryLayout<TintUniforms>.size,
      index: BufferIndex.tintUniforms.rawValue)
    encoder.setVertexBuffer(drawCommand.uniforms, offset: 0, index: BufferIndex.uniforms.rawValue)
    encoder.setVertexBuffer(buffer, offset: 0, index: BufferIndex.meshPositions.rawValue)

    let elapsed = -Float(viewStartTime.timeIntervalSinceNow)
    let dt = elapsed - frameDelta

    var params = Params(
      time: elapsed,
      elapsed: dt,
      viewerPosition: gestureManager.viewerPosition,
      viewerScale: gestureManager.viewerScale,
      viewerRotation: gestureManager.viewerRotation,
      totalLines: Int32(linesCount))

    let pb = device.makeBuffer(
      bytes: &params, length: MemoryLayout<Params>.stride,
      options: .storageModeShared)!
    encoder.setVertexBuffer(pb, offset: 0, index: BufferIndex.params.rawValue)
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
