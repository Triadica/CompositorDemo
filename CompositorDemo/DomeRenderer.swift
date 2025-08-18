import CompositorServices
import Metal
import MetalKit
import Spatial
import SwiftUI
import simd

private let maxFramesInFlight = 3

private let domeSegmentCount: Int = 10000  // 圆顶段数
private let segmentsPerDomeSegment: Int = 4  // 每个圆顶段的细分
private let verticesPerDomeSegment = segmentsPerDomeSegment * 2 + 2  // 上下底面各一个中心点
private let verticesCount = verticesPerDomeSegment * domeSegmentCount

private let sideIndexesPerDomeSegment: Int = 6 * segmentsPerDomeSegment  // 6 vertices per rectangle (side faces)
private let topIndexesPerDomeSegment: Int = segmentsPerDomeSegment * 3  // cover the top with triangles
private let bottomIndexesPerDomeSegment: Int = segmentsPerDomeSegment * 3  // cover the bottom with triangles
private let indexesPerDomeSegment =
  sideIndexesPerDomeSegment + topIndexesPerDomeSegment + bottomIndexesPerDomeSegment
private let indexesCount: Int = domeSegmentCount * indexesPerDomeSegment

private let domeSegmentHeight: Float = 0.04  // 圆顶段高度
private let domeSegmentRadius: Float = 0.01  // 圆顶段半径

private struct DomeSegmentBase {
  var position: SIMD3<Float>
  var color: SIMD3<Float>
  var segmentId: Float
  var velocity: SIMD3<Float> = .zero
  var activateTime: Float = 0.0  // 激活时间
  var isActive: Bool = false  // 是否激活
}

private struct Params {
  var viewerPosition: SIMD3<Float>
  var time: Float
  var viewerScale: Float
  var viewerRotation: Float = .zero
  var _padding: SIMD2<Float> = .zero  // required for 48 bytes alignment
}

@MainActor
class DomeRenderer: CustomRenderer {
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
    let cellUpdateBase = library.makeFunction(name: "domeComputeShader")!
    computePipeLine = try computeDevice.makeComputePipelineState(function: cellUpdateBase)

    computeCommandQueue = computeDevice.makeCommandQueue()!

    self.createDomeVerticesBuffer(device: layerRenderer.device)
    self.createDomeIndexBuffer(device: layerRenderer.device)
    self.createDomeComputeBuffer(device: layerRenderer.device)
  }

  /// create and sets the vertices of the dome segment
  private func createDomeVerticesBuffer(device: MTLDevice) {
    let bufferLength = MemoryLayout<LampsVertex>.stride * verticesCount
    vertexBuffer = device.makeBuffer(length: bufferLength)!
    vertexBuffer.label = "Dome segment vertex buffer"
    var cellVertices: UnsafeMutablePointer<LampsVertex> {
      vertexBuffer.contents().assumingMemoryBound(to: LampsVertex.self)
    }

    for i in 0..<domeSegmentCount {
      let baseIndex = i * verticesPerDomeSegment

      // Create dome segment vertices
      for s in 0..<segmentsPerDomeSegment {
        let angle = Float(s) * (2 * Float.pi / Float(segmentsPerDomeSegment))

        // Top edge
        let topEdge = SIMD3<Float>(
          cos(angle) * domeSegmentRadius, domeSegmentHeight / 2, sin(angle) * domeSegmentRadius)

        // Bottom edge
        let bottomEdge = SIMD3<Float>(
          cos(angle) * domeSegmentRadius, -domeSegmentHeight / 2, sin(angle) * domeSegmentRadius)

        // 橙色 (top)
        let orangeColor = SIMD3<Float>(1.0, 0.6, 0.2)
        // 红色 (bottom)
        let redColor = SIMD3<Float>(0.8, 0.2, 0.2)

        cellVertices[baseIndex + s] = LampsVertex(
          position: topEdge, color: orangeColor, seed: Int32(i))
        cellVertices[baseIndex + s + segmentsPerDomeSegment] = LampsVertex(
          position: bottomEdge, color: redColor, seed: Int32(i))
      }

      // Top center point (orange)
      cellVertices[baseIndex + segmentsPerDomeSegment * 2] = LampsVertex(
        position: SIMD3<Float>(0, domeSegmentHeight / 2, 0),
        color: SIMD3<Float>(1.0, 0.7, 0.3),
        seed: Int32(i)
      )

      // Bottom center point (red)
      cellVertices[baseIndex + segmentsPerDomeSegment * 2 + 1] = LampsVertex(
        position: SIMD3<Float>(0, -domeSegmentHeight / 2, 0),
        color: SIMD3<Float>(0.7, 0.1, 0.1),
        seed: Int32(i)
      )
    }
  }

  func resetComputeState() {
    self.createDomeComputeBuffer(device: computeDevice)
  }

  private func createDomeIndexBuffer(device: MTLDevice) {
    let bufferLength = MemoryLayout<UInt32>.stride * indexesCount
    indexBuffer = device.makeBuffer(length: bufferLength)!
    indexBuffer.label = "Dome segment index buffer"

    let cellIndices = indexBuffer.contents().bindMemory(
      to: UInt32.self, capacity: indexesCount)

    for i in 0..<domeSegmentCount {
      let verticesBase = i * verticesPerDomeSegment
      let indexBase = i * indexesPerDomeSegment

      // Side faces (rectangles)
      for s in 0..<segmentsPerDomeSegment {
        let topCurrent = verticesBase + s
        let topNext = verticesBase + (s + 1) % segmentsPerDomeSegment
        let bottomCurrent = verticesBase + s + segmentsPerDomeSegment
        let bottomNext = verticesBase + (s + 1) % segmentsPerDomeSegment + segmentsPerDomeSegment

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
      let topCenter = verticesBase + segmentsPerDomeSegment * 2
      let topIndexBase = indexBase + sideIndexesPerDomeSegment
      for s in 0..<segmentsPerDomeSegment {
        let current = verticesBase + s
        let next = verticesBase + (s + 1) % segmentsPerDomeSegment
        let triangleBase = topIndexBase + s * 3

        cellIndices[triangleBase] = UInt32(topCenter)
        cellIndices[triangleBase + 1] = UInt32(current)
        cellIndices[triangleBase + 2] = UInt32(next)
      }

      // Bottom cap
      let bottomCenter = verticesBase + segmentsPerDomeSegment * 2 + 1
      let bottomIndexBase = indexBase + sideIndexesPerDomeSegment + topIndexesPerDomeSegment
      for s in 0..<segmentsPerDomeSegment {
        let current = verticesBase + s + segmentsPerDomeSegment
        let next = verticesBase + (s + 1) % segmentsPerDomeSegment + segmentsPerDomeSegment
        let triangleBase = bottomIndexBase + s * 3

        cellIndices[triangleBase] = UInt32(bottomCenter)
        cellIndices[triangleBase + 1] = UInt32(next)
        cellIndices[triangleBase + 2] = UInt32(current)
      }
    }
  }

  private func createDomeComputeBuffer(device: MTLDevice) {
    let bufferLength = MemoryLayout<DomeSegmentBase>.stride * domeSegmentCount

    computeBuffer = PingPongBuffer(device: device, length: bufferLength)

    guard let computeBuffer = computeBuffer else {
      print("Failed to create compute buffer")
      return
    }
    computeBuffer.addLabel("Dome segment compute buffer")

    let contents = computeBuffer.currentBuffer.contents()
    let domeSegmentBase = contents.bindMemory(to: DomeSegmentBase.self, capacity: domeSegmentCount)

    for i in 0..<domeSegmentCount {
      // 形成圆顶状分布
      let angle = Float(i) * (2 * Float.pi / Float(domeSegmentCount))
      let radius = Float.random(in: 5...20)
      let height = sqrt(max(0, 400 - radius * radius)) / 10 // 圆顶高度函数
      
      let xOffset = cos(angle) * radius
      let zOffset = sin(angle) * radius
      let yOffset = height

      let domeSegmentPosition = SIMD3<Float>(xOffset, yOffset, zOffset)

      // 圆顶段颜色 (橙红渐变)
      let color = SIMD3<Float>(1.0, 0.5, 0.2)

      // 静态或缓慢移动
      let velocity = SIMD3<Float>(
        Float.random(in: (-0.1)...0.1),
        Float.random(in: (-0.1)...0.1),
        Float.random(in: (-0.1)...0.1)
      )

      domeSegmentBase[i] = DomeSegmentBase(
        position: domeSegmentPosition,
        color: color,
        segmentId: Float(i),
        velocity: velocity,
        activateTime: 0.0,
        isActive: true
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
      MemoryLayout<LampsVertex>.stride
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

    let vertexFunction = library.makeFunction(name: "domeVertexShader")
    let fragmentFunction = library.makeFunction(name: "domeFragmentShader")

    pipelineDescriptor.fragmentFunction = fragmentFunction
    pipelineDescriptor.vertexFunction = vertexFunction

    pipelineDescriptor.label = "DomeRenderPipeline"
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
      width: (domeSegmentCount + threadGroupSize - 1) / threadGroupSize,
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

    // let bufferLength = MemoryLayout<LampsVertex>.stride * numVertices

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
