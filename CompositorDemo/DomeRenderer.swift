import CompositorServices
import Metal
import MetalKit
import Spatial
import SwiftUI
import simd

private let maxFramesInFlight = 3

// 球壳参数
private let sphereRadius: Float = 5.0  // 球壳半径 5m
private let pointCount: Int = 40  // 球壳上的点数量

// 球壳网格参数 - 大幅减少密度以提升性能
private let sphereSegments: Int = 16  // 球壳经度分段
private let sphereRings: Int = 8  // 球壳纬度分段
private let verticesCount = (sphereRings + 1) * (sphereSegments + 1)
private let indexesCount = sphereRings * sphereSegments * 6

private struct SpherePoint {
  var position: SIMD3<Float>  // 球壳上的点位置
  var velocity: SIMD3<Float>  // 移动速度
  var pointId: Float  // 点ID
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

  /// 创建球壳顶点缓冲区
  private func createDomeVerticesBuffer(device: MTLDevice) {
    let bufferLength = MemoryLayout<VertexWithSeed>.stride * verticesCount
    vertexBuffer = device.makeBuffer(length: bufferLength)!
    vertexBuffer.label = "Sphere vertex buffer"

    let vertices = vertexBuffer.contents().assumingMemoryBound(to: VertexWithSeed.self)

    var vertexIndex = 0

    // 生成球壳顶点
    for ring in 0...sphereRings {
      let phi = Float(ring) * Float.pi / Float(sphereRings)  // 纬度角
      let y = cos(phi) * sphereRadius
      let ringRadius = sin(phi) * sphereRadius

      for segment in 0...sphereSegments {
        let theta = Float(segment) * 2.0 * Float.pi / Float(sphereSegments)  // 经度角
        let x = cos(theta) * ringRadius
        let z = sin(theta) * ringRadius

        let position = SIMD3<Float>(x, y, z)
        let color = SIMD3<Float>(0.2, 0.2, 0.2)  // 默认灰色

        vertices[vertexIndex] = VertexWithSeed(
          position: position,
          color: color,
          seed: Int32(vertexIndex)
        )
        vertexIndex += 1

      }
    }
  }

  func resetComputeState() {
    self.createDomeComputeBuffer(device: self.computeDevice)
  }

  /// 创建球壳索引缓冲区
  private func createDomeIndexBuffer(device: MTLDevice) {
    let bufferLength = MemoryLayout<UInt32>.stride * indexesCount
    indexBuffer = device.makeBuffer(length: bufferLength)!
    indexBuffer.label = "Sphere index buffer"

    let indices = indexBuffer.contents().assumingMemoryBound(to: UInt32.self)
    var indexOffset = 0

    // 生成球壳三角形索引
    for ring in 0..<sphereRings {
      for segment in 0..<sphereSegments {
        let current = UInt32(ring * (sphereSegments + 1) + segment)
        let next = UInt32(ring * (sphereSegments + 1) + (segment + 1))
        let currentNext = UInt32((ring + 1) * (sphereSegments + 1) + segment)
        let nextNext = UInt32((ring + 1) * (sphereSegments + 1) + (segment + 1))

        // 第一个三角形
        indices[indexOffset] = current
        indices[indexOffset + 1] = currentNext
        indices[indexOffset + 2] = next

        // 第二个三角形
        indices[indexOffset + 3] = next
        indices[indexOffset + 4] = currentNext
        indices[indexOffset + 5] = nextNext

        indexOffset += 6
      }
    }
  }

  /// 创建球壳上的点数据
  private func createDomeComputeBuffer(device: MTLDevice) {
    let bufferLength = MemoryLayout<SpherePoint>.stride * pointCount

    computeBuffer = PingPongBuffer(device: device, length: bufferLength)

    guard let computeBuffer = computeBuffer else {
      print("Failed to create compute buffer")
      return
    }
    computeBuffer.addLabel("Sphere points compute buffer")

    let contents = computeBuffer.currentBuffer.contents()
    let spherePoints = contents.bindMemory(to: SpherePoint.self, capacity: pointCount)

    for i in 0..<pointCount {
      // 在球壳上随机分布点
      let phi = Float.random(in: 0...Float.pi)  // 纬度角
      let theta = Float.random(in: 0...(2 * Float.pi))  // 经度角

      let x = sin(phi) * cos(theta) * sphereRadius
      let y = cos(phi) * sphereRadius
      let z = sin(phi) * sin(theta) * sphereRadius

      let position = SIMD3<Float>(x, y, z)

      // 缓慢的随机移动速度
      let velocity = SIMD3<Float>(
        Float.random(in: -0.02...0.02),
        Float.random(in: -0.02...0.02),
        Float.random(in: -0.02...0.02)
      )

      spherePoints[i] = SpherePoint(
        position: position,
        velocity: velocity,
        pointId: Float(i)
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
    // let dt = delta - frameDelta
    frameDelta = delta

    var params = Params(
      viewerPosition: gestureManager.viewerPosition,
      time: 0.016,  // 使用固定的时间步长，避免闪烁
      viewerScale: gestureManager.viewerScale,
      viewerRotation: gestureManager.viewerRotation
    )
    computeEncoder.setBytes(&params, length: MemoryLayout<Params>.size, index: 2)
    let threadGroupSize = min(computePipeLine.maxTotalThreadsPerThreadgroup, 256)
    let threadsPerThreadgroup = MTLSize(width: threadGroupSize, height: 1, depth: 1)
    let threadGroups = MTLSize(
      width: (pointCount + threadGroupSize - 1) / threadGroupSize,
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
      time: 0.016,  // 使用固定的时间步长，避免闪烁
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

    // 为 fragment shader 设置缓冲区
    encoder.setFragmentBuffer(
      params,
      offset: 0,
      index: BufferIndex.params.rawValue)

    encoder.setFragmentBuffer(
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
