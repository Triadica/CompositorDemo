/*
 See the LICENSE.txt file for this sample's licensing information.

 Abstract:
 A renderer that displays a set of color swatches.
 */

import Metal
import MetalKit
import Spatial
import SwiftUI
import simd

#if canImport(CompositorServices)
  import CompositorServices
#endif

private let maxFramesInFlight = 3

// 桃花相关常量
// 性能优化：使用编译时常量减少运行时计算
private let flowerCount: Int = 12  // 增加花朵数量 (3 -> 12)
private let petalsPerFlower: Int = 6  // 每朵花6个花瓣

// 与 Metal 里的常量同步
private let kRadialSegments = 32
private let kLateralSegments = 16
private let kVerticesPerPetal = kRadialSegments * kLateralSegments * 6

// 预计算的常量
private let totalPetals = flowerCount * petalsPerFlower
private let verticesCount = totalPetals * kVerticesPerPetal

// 花朵尺寸常量
private let flowerSize: Float = 0.3
private let flowerRadius: Float = 0.15
private let petalLength: Float = 0.12  // 适当增大基础尺度
private let petalWidth: Float = 0.08

private struct CellBase {
  var position: SIMD3<Float>  // 线段在花朵内的相对位置
  var color: SIMD3<Float>  // 线段颜色
  var flowerId: Float  // 所属花朵的ID
  var flowerCenter: SIMD3<Float>  // 花朵中心位置
  var petalId: Float  // 所属花瓣的ID
  var lineType: Float  // 线条类型：0=轮廓线，1=填充线
  var petalAngle: Float  // 花瓣在花朵中的角度
  var petalSize: Float = 0.08  // 花瓣大小
}

private struct Params {
  var viewerPosition: SIMD3<Float>
  var time: Float
  var viewerScale: Float
  var viewerRotation: Float = .zero
  var _padding: SIMD4<Float> = .zero  // offset 32 (alignment 16), total = 48 bytes to match Metal float3 struct padding
}

@MainActor
class FlowersRenderer: CustomRenderer {
  private let renderPipelineState: MTLRenderPipelineState & Sendable

  private var uniformsBuffer: [MTLBuffer]
  /// a buffer to hold the vertices of the lamp
  var vertexBuffer: MTLBuffer!

  var indexBuffer: MTLBuffer!
  var petalDataBuffer: MTLBuffer!  // 存储花瓣数据的缓冲区

  // 性能优化：预分配参数缓冲区，避免每帧创建
  private var paramsBuffer: MTLBuffer!

  var gestureManager: GestureManager = GestureManager(onScene: false)
  nonisolated var usesFoveation: Bool { false }

  init(layerRenderer: LayerRenderer) throws {
    uniformsBuffer = (0..<Renderer.maxFramesInFlight).map { _ in
      layerRenderer.device.makeBuffer(length: MemoryLayout<PathProperties>.uniformStride)!
    }

    // 性能优化：预分配参数缓冲区
    paramsBuffer = layerRenderer.device.makeBuffer(
      length: MemoryLayout<Params>.size,
      options: .storageModeShared
    )!

    renderPipelineState = try Self.makeRenderPipelineDescriptor(layerRenderer: layerRenderer)

    self.createFlowerVerticesBuffer(device: layerRenderer.device)
    self.createFlowerIndexBuffer(device: layerRenderer.device)
    self.createPetalDataBuffer(device: layerRenderer.device)
  }

  /// 创建花朵的花瓣顶点（使用线段构建花瓣轮廓和填充）
  private func createFlowerVerticesBuffer(device: MTLDevice) {
    let bufferLength = MemoryLayout<VertexWithSeed>.stride * verticesCount
    vertexBuffer = device.makeBuffer(length: bufferLength)!
    vertexBuffer.label = "Flower petal vertex buffer"
    var cellVertices: UnsafeMutablePointer<VertexWithSeed> {
      vertexBuffer.contents().assumingMemoryBound(to: VertexWithSeed.self)
    }

    // 我们只需要填入 seed，Shader 会根据 vid(0...kVerticesPerPetal-1) 和 seed 自动生成网格
    var vCount = 0
    for flowerId in 0..<flowerCount {
      for petalId in 0..<petalsPerFlower {
        let seed = Int32(flowerId * petalsPerFlower + petalId)
        for _ in 0..<kVerticesPerPetal {
          cellVertices[vCount] = VertexWithSeed(
            position: .zero,
            color: .zero,
            seed: seed
          )
          vCount += 1
        }
      }
    }
  }

  private func createFlowerIndexBuffer(device: MTLDevice) {
    let bufferLength = MemoryLayout<UInt32>.stride * verticesCount
    indexBuffer = device.makeBuffer(length: bufferLength)!
    indexBuffer.label = "Flower petal index buffer"

    let cellIndices = indexBuffer.contents().bindMemory(
      to: UInt32.self, capacity: verticesCount)

    // 简单的 1:1 映射
    for i in 0..<verticesCount {
      cellIndices[i] = UInt32(i)
    }
  }

  private func createPetalDataBuffer(device: MTLDevice) {
    let bufferSize = MemoryLayout<CellBase>.stride * totalPetals
    petalDataBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared)!

    let petalBase = petalDataBuffer.contents().bindMemory(
      to: CellBase.self, capacity: totalPetals)

    // 基础颜色主题
    let baseColors: [SIMD3<Float>] = [
      SIMD3<Float>(1.0, 0.3, 0.4),  // 粉红色
      SIMD3<Float>(0.9, 0.7, 0.2),  // 金黄色
      SIMD3<Float>(0.7, 0.4, 0.9),  // 紫色
      SIMD3<Float>(1.0, 0.5, 0.2),  // 橙色
      SIMD3<Float>(1.0, 0.1, 0.6),  // 深粉
    ]

    var petalIndex = 0

    for flowerId in 0..<flowerCount {
      let flowerColor = baseColors[flowerId % baseColors.count]

      // 在 3D 空间内生成更广泛的分布
      // x: -2.5 ~ 2.5, y: -0.5 ~ 1.5, z: -2.0 ~ -5.5
      let seed = Float(flowerId)
      let randX = sin(seed * 0.5) * 2.2 + cos(seed * 0.2) * 0.5
      let randY = cos(seed * 0.8) * 0.8 + 0.5
      let randZ = -2.5 - abs(sin(seed * 1.3)) * 3.5
      let flowerCenter = SIMD3<Float>(randX, randY, randZ)

      for petalId in 0..<petalsPerFlower {
        let petalAngle = Float(petalId) * (2.0 * Float.pi / Float(petalsPerFlower))
        let petalSize = Float.random(in: petalWidth...petalLength)

        // 每个花瓣颜色微调
        let colorVariation = Float.random(in: 0.94...1.06)
        let variedColor = flowerColor * colorVariation

        petalBase[petalIndex] = CellBase(
          position: .zero,
          color: variedColor,
          flowerId: Float(flowerId),
          flowerCenter: flowerCenter,
          petalId: Float(petalId),
          lineType: 0.0,
          petalAngle: petalAngle,
          petalSize: petalSize
        )

        petalIndex += 1
      }
    }
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

    let vertexFunction = library.makeFunction(name: "flowersVertexShader")
    let fragmentFunction = library.makeFunction(name: "flowersFragmentShader")

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

  // 移除compute相关函数

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

    // 性能优化：使用预分配的缓冲区，避免每帧创建
    let params_data = Params(
      viewerPosition: gestureManager.viewerPosition,
      time: getTimeSinceStart(),
      viewerScale: gestureManager.viewerScale,
      viewerRotation: gestureManager.viewerRotation
    )

    // 直接更新预分配缓冲区的内容
    let paramsPointer = paramsBuffer.contents().bindMemory(to: Params.self, capacity: 1)
    paramsPointer.pointee = params_data

    encoder.setVertexBuffer(
      paramsBuffer,
      offset: 0,
      index: BufferIndex.params.rawValue)

    encoder.setVertexBuffer(
      petalDataBuffer, offset: 0, index: BufferIndex.base.rawValue)

    encoder.drawIndexedPrimitives(
      type: .triangle,
      indexCount: verticesCount,
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

  // 协议要求的方法，但由于已移除compute shader，这些方法为空实现
  func resetComputeState() {
    // 不再需要compute state重置
  }

  func computeCommandCommit() {
    // 不再需要compute command提交
  }
}
