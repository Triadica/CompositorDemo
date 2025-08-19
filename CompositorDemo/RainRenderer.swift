import CompositorServices
import Metal
import MetalKit
import Spatial
import SwiftUI
import simd

private let maxFramesInFlight = 3

private let raindropCount: Int = 10000  // 增加到10000个雨滴
private let segmentsPerRaindrop: Int = 4  // 进一步简化到4段，减少三角形数量
private let verticesPerRaindrop = segmentsPerRaindrop * 2 + 2  // 上下底面各一个中心点
private let verticesCount = verticesPerRaindrop * raindropCount

private let sideIndexesPerRaindrop: Int = 6 * segmentsPerRaindrop  // 6 vertices per rectangle (side faces)
private let topIndexesPerRaindrop: Int = segmentsPerRaindrop * 3  // cover the top with triangles
private let bottomIndexesPerRaindrop: Int = segmentsPerRaindrop * 3  // cover the bottom with triangles
private let indexesPerRaindrop =
  sideIndexesPerRaindrop + topIndexesPerRaindrop + bottomIndexesPerRaindrop
private let indexesCount: Int = raindropCount * indexesPerRaindrop

private let raindropHeight: Float = 0.04  // 更小的雨滴尺寸
private let raindropRadius: Float = 0.01  // 更小的雨滴半径

private struct RaindropBase {
  var position: SIMD3<Float>
  var color: SIMD3<Float>
  var raindropId: Float
  var velocity: SIMD3<Float> = .zero
  var groundTime: Float = 0.0  // Time staying on the ground
  var isOnGround: Bool = false  // Whether on the ground
}

private struct Params {
  var viewerPosition: SIMD3<Float>
  var time: Float
  var viewerScale: Float
  var viewerRotation: Float = .zero
  var _padding: SIMD2<Float> = .zero  // required for 48 bytes alignment
}

@MainActor
class RainRenderer: CustomRenderer {
  private let renderPipelineState: MTLRenderPipelineState & Sendable

  private var uniformsBuffer: [MTLBuffer]
  /// a buffer to hold the vertices of the lamp
  var vertexBuffer: MTLBuffer!

  var indexBuffer: MTLBuffer!

  let computeDevice: MTLDevice
  var computeBuffer: PingPongBuffer?
  let computePipeLine: MTLComputePipelineState
  let computeCommandQueue: MTLCommandQueue

  var gestureManager: GestureManager = GestureManager(onScene: true)

  init(layerRenderer: LayerRenderer) throws {
    uniformsBuffer = (0..<Renderer.maxFramesInFlight).map { _ in
      layerRenderer.device.makeBuffer(length: MemoryLayout<PathProperties>.uniformStride)!
    }

    renderPipelineState = try Self.makeRenderPipelineDescriptor(layerRenderer: layerRenderer)

    self.computeDevice = MTLCreateSystemDefaultDevice()!
    let library = computeDevice.makeDefaultLibrary()!
    let cellUpdateBase = library.makeFunction(name: "rainComputeShader")!
    computePipeLine = try computeDevice.makeComputePipelineState(function: cellUpdateBase)

    computeCommandQueue = computeDevice.makeCommandQueue()!

    self.createLampVerticesBuffer(device: layerRenderer.device)
    self.createLampIndexBuffer(device: layerRenderer.device)
    self.createLampComputeBuffer(device: layerRenderer.device)
  }

  /// create and sets the vertices of the raindrop
  private func createLampVerticesBuffer(device: MTLDevice) {
    let bufferLength = MemoryLayout<VertexWithSeed>.stride * verticesCount
    vertexBuffer = device.makeBuffer(length: bufferLength)!
    vertexBuffer.label = "Raindrop vertex buffer"
    var cellVertices: UnsafeMutablePointer<VertexWithSeed> {
      vertexBuffer.contents().assumingMemoryBound(to: VertexWithSeed.self)
    }

    for i in 0..<raindropCount {
      let baseIndex = i * verticesPerRaindrop

      // Create raindrop vertices
      for s in 0..<segmentsPerRaindrop {
        let angle = Float(s) * (2 * Float.pi / Float(segmentsPerRaindrop))

        // Top edge (white)
        let topEdge = SIMD3<Float>(
          cos(angle) * raindropRadius, raindropHeight / 2, sin(angle) * raindropRadius)

        // Bottom edge (blue)
        let bottomEdge = SIMD3<Float>(
          cos(angle) * raindropRadius, -raindropHeight / 2, sin(angle) * raindropRadius)

        // White color (top)
        let whiteColor = SIMD3<Float>(0.9, 0.9, 1.0)
        // Blue color (bottom)
        let blueColor = SIMD3<Float>(0.2, 0.4, 0.8)

        cellVertices[baseIndex + s] = VertexWithSeed(
          position: topEdge, color: whiteColor, seed: Int32(i))
        cellVertices[baseIndex + s + segmentsPerRaindrop] = VertexWithSeed(
          position: bottomEdge, color: blueColor, seed: Int32(i))
      }

      // Top center point (white)
      cellVertices[baseIndex + segmentsPerRaindrop * 2] = VertexWithSeed(
        position: SIMD3<Float>(0, raindropHeight / 2, 0),
        color: SIMD3<Float>(1.0, 1.0, 1.0),
        seed: Int32(i)
      )

      // Bottom center point (blue)
      cellVertices[baseIndex + segmentsPerRaindrop * 2 + 1] = VertexWithSeed(
        position: SIMD3<Float>(0, -raindropHeight / 2, 0),
        color: SIMD3<Float>(0.1, 0.2, 0.6),
        seed: Int32(i)
      )
    }
  }

  func resetComputeState() {
    self.createLampComputeBuffer(device: computeDevice)
  }

  private func createLampIndexBuffer(device: MTLDevice) {
    let bufferLength = MemoryLayout<UInt32>.stride * indexesCount
    indexBuffer = device.makeBuffer(length: bufferLength)!
    indexBuffer.label = "Raindrop index buffer"

    let cellIndices = indexBuffer.contents().bindMemory(
      to: UInt32.self, capacity: indexesCount)

    for i in 0..<raindropCount {
      let verticesBase = i * verticesPerRaindrop
      let indexBase = i * indexesPerRaindrop

      // Side faces (rectangles)
      for s in 0..<segmentsPerRaindrop {
        let topCurrent = verticesBase + s
        let topNext = verticesBase + (s + 1) % segmentsPerRaindrop
        let bottomCurrent = verticesBase + s + segmentsPerRaindrop
        let bottomNext = verticesBase + (s + 1) % segmentsPerRaindrop + segmentsPerRaindrop

        let sideIndexBase = indexBase + s * 6

        // First triangle
        cellIndices[sideIndexBase] = UInt32(topCurrent)
        cellIndices[sideIndexBase + 1] = UInt32(bottomCurrent)
        cellIndices[sideIndexBase + 2] = UInt32(topNext)

        // Second triangle
        cellIndices[sideIndexBase + 3] = UInt32(topNext)
        cellIndices[sideIndexBase + 4] = UInt32(bottomCurrent)
        cellIndices[sideIndexBase + 5] = UInt32(bottomNext)
      }

      // Top cap
      let topCenter = verticesBase + segmentsPerRaindrop * 2
      let topIndexBase = indexBase + sideIndexesPerRaindrop
      for s in 0..<segmentsPerRaindrop {
        let current = verticesBase + s
        let next = verticesBase + (s + 1) % segmentsPerRaindrop
        let triangleBase = topIndexBase + s * 3

        cellIndices[triangleBase] = UInt32(topCenter)
        cellIndices[triangleBase + 1] = UInt32(current)
        cellIndices[triangleBase + 2] = UInt32(next)
      }

      // Bottom cap
      let bottomCenter = verticesBase + segmentsPerRaindrop * 2 + 1
      let bottomIndexBase = indexBase + sideIndexesPerRaindrop + topIndexesPerRaindrop
      for s in 0..<segmentsPerRaindrop {
        let current = verticesBase + s + segmentsPerRaindrop
        let next = verticesBase + (s + 1) % segmentsPerRaindrop + segmentsPerRaindrop
        let triangleBase = bottomIndexBase + s * 3

        cellIndices[triangleBase] = UInt32(bottomCenter)
        cellIndices[triangleBase + 1] = UInt32(next)
        cellIndices[triangleBase + 2] = UInt32(current)
      }
    }
  }

  private func createLampComputeBuffer(device: MTLDevice) {
    let bufferLength = MemoryLayout<RaindropBase>.stride * raindropCount

    computeBuffer = PingPongBuffer(device: device, length: bufferLength)

    guard let computeBuffer = computeBuffer else {
      print("Failed to create compute buffer")
      return
    }
    computeBuffer.addLabel("Raindrop compute buffer")

    let contents = computeBuffer.currentBuffer.contents()
    let raindropBase = contents.bindMemory(to: RaindropBase.self, capacity: raindropCount)

    for i in 0..<raindropCount {
      // Randomly distributed in the sky
      let xOffset = Float.random(in: (-25)...25)
      let zOffset = Float.random(in: (-30)...30)
      let yOffset = Float.random(in: 8...15)  // Start from high altitude

      let raindropPosition = SIMD3<Float>(xOffset, yOffset, zOffset)

      // Raindrop color (white to blue gradient, base color set here)
      let color = SIMD3<Float>(0.6, 0.7, 0.9)

      // Falling velocity
      let minY: Float = -2.0
      let maxY: Float = -1.2
      let velocity = SIMD3<Float>(
        Float.random(in: (-0.2)...0.2),  // Slight horizontal drift
        Float.random(in: minY...maxY),  // Main downward velocity
        Float.random(in: (-0.2)...0.2)  // Slight horizontal drift
      )

      raindropBase[i] = RaindropBase(
        position: raindropPosition,
        color: color,
        raindropId: Float(i),
        velocity: velocity,
        groundTime: 0.0,
        isOnGround: false
      )
    }

    computeBuffer.copyToNext()
  }

  class func buildMetalVertexDescriptor() -> MTLVertexDescriptor {
    // Create a vertex descriptor specifying how Metal lays out vertices for input into the render pipeline.

    let mtlVertexDescriptor = MTLVertexDescriptor()

    mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].format =
      MTLVertexFormat.float3
    mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].offset = 0
    mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].bufferIndex =
      BufferIndex.meshPositions.rawValue

    let offset = MemoryLayout<SIMD3<Float>>.stride
    mtlVertexDescriptor.attributes[VertexAttribute.color.rawValue].format =
      MTLVertexFormat.float3
    mtlVertexDescriptor.attributes[VertexAttribute.color.rawValue].offset = offset
    mtlVertexDescriptor.attributes[VertexAttribute.color.rawValue].bufferIndex =
      BufferIndex.meshPositions.rawValue

    mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stride =
      MemoryLayout<VertexWithSeed>.stride
    mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepRate = 1
    mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepFunction =
      MTLVertexStepFunction.perVertex
    // add params for seed value
    let nextOffset = offset + MemoryLayout<SIMD3<Float>>.stride
    mtlVertexDescriptor.attributes[VertexAttribute.seed.rawValue].format =
      MTLVertexFormat.int
    mtlVertexDescriptor.attributes[VertexAttribute.seed.rawValue].offset = nextOffset
    mtlVertexDescriptor.attributes[VertexAttribute.seed.rawValue].bufferIndex =
      BufferIndex.meshPositions.rawValue

    return mtlVertexDescriptor
  }

  private static func makeRenderPipelineDescriptor(layerRenderer: LayerRenderer) throws
    -> MTLRenderPipelineState
  {
    let pipelineDescriptor = Renderer.defaultRenderPipelineDescriptor(
      layerRenderer: layerRenderer)

    let library = layerRenderer.device.makeDefaultLibrary()!

    let vertexFunction = library.makeFunction(name: "rainVertexShader")
    let fragmentFunction = library.makeFunction(name: "rainFragmentShader")

    pipelineDescriptor.fragmentFunction = fragmentFunction
    pipelineDescriptor.vertexFunction = vertexFunction

    pipelineDescriptor.label = "TriangleRenderPipeline"
    pipelineDescriptor.vertexDescriptor = self.buildMetalVertexDescriptor()

    return try layerRenderer.device.makeRenderPipelineState(descriptor: pipelineDescriptor)
  }

  func drawCommand(frame: LayerRenderer.Frame) throws -> TintDrawCommand {
    return TintDrawCommand(
      frameIndex: frame.frameIndex,
      uniforms: self.uniformsBuffer[Int(frame.frameIndex % Renderer.maxFramesInFlight)],
      verticesCount: verticesCount)
  }

  func computeCommandCommit() {
    guard let computeBuffer: PingPongBuffer = computeBuffer,
      let commandBuffer = computeCommandQueue.makeCommandBuffer(),
      let computeEncoder = commandBuffer.makeComputeCommandEncoder()
    else {
      print("Failed to create compute command buffer")
      return
    }

    computeEncoder.setComputePipelineState(computePipeLine)
    computeEncoder.setBuffer(computeBuffer.currentBuffer, offset: 0, index: 0)
    computeEncoder.setBuffer(computeBuffer.nextBuffer, offset: 0, index: 1)

    let delta = -Float(viewStartTime.timeIntervalSinceNow)
    let dt = delta - frameDelta
    frameDelta = delta

    var params = Params(
      viewerPosition: gestureManager.viewerPosition,
      time: dt,
      viewerScale: gestureManager.viewerScale,
      viewerRotation: gestureManager.viewerRotation
    )
    computeEncoder.setBytes(&params, length: MemoryLayout<Params>.size, index: 2)
    let threadGroupSize = min(computePipeLine.maxTotalThreadsPerThreadgroup, 256)
    let threadsPerThreadgroup = MTLSize(width: threadGroupSize, height: 1, depth: 1)
    let threadGroups = MTLSize(
      width: (raindropCount + threadGroupSize - 1) / threadGroupSize,
      height: 1,
      depth: 1
    )
    computeEncoder.dispatchThreadgroups(
      threadGroups, threadsPerThreadgroup: threadsPerThreadgroup)
    computeEncoder.endEncoding()
    commandBuffer.commit()

    computeBuffer.swap()
  }

  // in seconds
  func getTimeSinceStart() -> Float {
    let time = DispatchTime.now().uptimeNanoseconds
    let timeSinceStart = Float(time) / 1_000_000_000
    return timeSinceStart
  }

  private var viewStartTime: Date = Date()
  private var frameDelta: Float = 0.0

  func encodeDraw(
    _ drawCommand: TintDrawCommand,
    encoder: MTLRenderCommandEncoder,
    drawable: LayerRenderer.Drawable,
    device: MTLDevice, tintValue: Float,
    buffer: MTLBuffer,
    indexBuffer: MTLBuffer
  ) {
    encoder.setCullMode(.none)

    encoder.setRenderPipelineState(renderPipelineState)

    var demoUniform: TintUniforms = TintUniforms(tintOpacity: tintValue)
    encoder.setVertexBytes(
      &demoUniform,
      length: MemoryLayout<TintUniforms>.size,
      index: BufferIndex.tintUniforms.rawValue)

    encoder.setVertexBuffer(
      drawCommand.uniforms,
      offset: 0,
      index: BufferIndex.uniforms.rawValue)

    encoder.setVertexBuffer(
      buffer,
      offset: 0,
      index: BufferIndex.meshPositions.rawValue)

    var params_data = Params(
      viewerPosition: gestureManager.viewerPosition,
      time: frameDelta,  // Use the same delta time as compute shader
      viewerScale: gestureManager.viewerScale,
      viewerRotation: gestureManager.viewerRotation
    )

    let params: any MTLBuffer = device.makeBuffer(
      bytes: &params_data,
      length: MemoryLayout<Params>.size,
      options: .storageModeShared
    )!

    encoder.setVertexBuffer(
      params,
      offset: 0,
      index: BufferIndex.params.rawValue)

    encoder.setVertexBuffer(
      computeBuffer?.currentBuffer, offset: 0, index: BufferIndex.base.rawValue)

    encoder.drawIndexedPrimitives(
      type: .triangle,
      indexCount: indexesCount,
      indexType: .uint32,
      indexBuffer: indexBuffer,
      indexBufferOffset: 0
    )
  }

  func updateUniformBuffers(
    _ drawCommand: TintDrawCommand,
    drawable: LayerRenderer.Drawable
  ) {
    drawCommand.uniforms.contents().assumingMemoryBound(to: Uniforms.self).pointee = Uniforms(
      drawable: drawable)
  }

  func onSpatialEvents(events: SpatialEventCollection) {
    for event in events {
      gestureManager.onSpatialEvent(event: event)
    }
  }
}
