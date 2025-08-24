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

  // 花朵相关常量
private let flowerCount: Int = 2  // 简化为2朵花
private let petalPerFlower: Int = 2000  // 每朵花的花瓣数量，优化性能
private let totalPetals = flowerCount * petalPerFlower

  // 花瓣顶点数（每个花瓣3个顶点）
private let verticesPerPetal = 3
private let verticesCount = totalPetals * verticesPerPetal

  // 索引数（每个花瓣3个索引）
private let indexesPerPetal = 3
private let indexesCount = totalPetals * indexesPerPetal

  // 花朵尺寸常量
private let flowerSize: Float = 0.3  // 花朵直径约0.3米
private let petalMinSize: Float = 0.00075
let petalMaxSize: Float = 0.00125  // 极微小花瓣尺寸
private let flowerRadius: Float = flowerSize * 0.5  // 花朵半径

private struct CellBase {
  var position: SIMD3<Float>  // 花瓣在花朵内的相对位置
  var color: SIMD3<Float>  // 花瓣颜色（与所属花朵一致）
  var flowerId: Float  // 所属花朵的ID
  var flowerCenter: SIMD3<Float>  // 花朵中心位置
  var rotationAngle: Float = 0.0  // 花朵当前旋转角度
  var petalSize: Float = 0.01  // 花瓣大小
}

private struct Params {
  var viewerPosition: SIMD3<Float>
  var time: Float
  var viewerScale: Float
  var viewerRotation: Float = .zero
  var _padding: SIMD2<Float> = .zero  // required for 48 bytes alignment
}

@MainActor
class FlowersRenderer: CustomRenderer {
  private let renderPipelineState: MTLRenderPipelineState & Sendable
  
  private var uniformsBuffer: [MTLBuffer]
    /// a buffer to hold the vertices of the lamp
  var vertexBuffer: MTLBuffer!
  
  var indexBuffer: MTLBuffer!
  var petalDataBuffer: MTLBuffer!  // 存储花瓣数据的缓冲区
  
  var gestureManager: GestureManager = GestureManager(onScene: false)
  
  init(layerRenderer: LayerRenderer) throws {
    uniformsBuffer = (0..<Renderer.maxFramesInFlight).map { _ in
      layerRenderer.device.makeBuffer(length: MemoryLayout<PathProperties>.uniformStride)!
    }
    
    renderPipelineState = try Self.makeRenderPipelineDescriptor(layerRenderer: layerRenderer)
    
    self.createFlowerVerticesBuffer(device: layerRenderer.device)
    self.createFlowerIndexBuffer(device: layerRenderer.device)
    self.createPetalDataBuffer(device: layerRenderer.device)
  }
  
    /// 创建正八面体区域内的小三角形顶点
  private func createFlowerVerticesBuffer(device: MTLDevice) {
    let bufferLength = MemoryLayout<VertexWithSeed>.stride * verticesCount
    vertexBuffer = device.makeBuffer(length: bufferLength)!
    vertexBuffer.label = "Flower petal vertex buffer"
    var cellVertices: UnsafeMutablePointer<VertexWithSeed> {
      vertexBuffer.contents().assumingMemoryBound(to: VertexWithSeed.self)
    }
    
    var vertexIndex = 0
    
      // 为每个正八面体生成多个小三角形
    for flowerId in 0..<flowerCount {
      for petalId in 0..<petalPerFlower {
          // 在花朵区域内随机生成花瓣位置
        let randomPosition = generateRandomPositionInFlower()
        
          // 生成小花瓣的三个顶点
        let petalSize = Float.random(in: petalMinSize...petalMaxSize)
        let height = petalSize * sqrt(3.0) / 2.0
        
          // 生成随机3D旋转角度
        let rotationX = Float.random(in: 0...(2 * Float.pi))
        let rotationY = Float.random(in: 0...(2 * Float.pi))
        let rotationZ = Float.random(in: 0...(2 * Float.pi))
        
          // 创建旋转矩阵
        let cosX = cos(rotationX)
        let sinX = sin(rotationX)
        let cosY = cos(rotationY)
        let sinY = sin(rotationY)
        let cosZ = cos(rotationZ)
        let sinZ = sin(rotationZ)
        
          // 组合旋转矩阵 (Z * Y * X)
        let rotationMatrix = simd_float3x3(
          SIMD3<Float>(cosY * cosZ, -cosY * sinZ, sinY),
          SIMD3<Float>(
            sinX * sinY * cosZ + cosX * sinZ, -sinX * sinY * sinZ + cosX * cosZ, -sinX * cosY),
          SIMD3<Float>(
            -cosX * sinY * cosZ + sinX * sinZ, cosX * sinY * sinZ + sinX * cosZ, cosX * cosY)
        )
        
          // 花瓣的三个顶点（相对于花瓣中心）
        let vertex1 = SIMD3<Float>(0.0, height / 3.0, 0.0)
        let vertex2 = SIMD3<Float>(-petalSize / 2.0, -height * 2.0 / 3.0, 0.0)
        let vertex3 = SIMD3<Float>(petalSize / 2.0, -height * 2.0 / 3.0, 0.0)
        
          // 应用随机3D旋转
        let rotatedVertex1 = rotationMatrix * vertex1
        let rotatedVertex2 = rotationMatrix * vertex2
        let rotatedVertex3 = rotationMatrix * vertex3
        
          // 添加随机位置偏移
        let finalVertex1 = rotatedVertex1 + randomPosition
        let finalVertex2 = rotatedVertex2 + randomPosition
        let finalVertex3 = rotatedVertex3 + randomPosition
        
        let petalIndex = flowerId * petalPerFlower + petalId
        
          // 创建顶点
        cellVertices[vertexIndex] = VertexWithSeed(
          position: finalVertex1,
          color: SIMD3<Float>(1.0, 1.0, 1.0),
          seed: Int32(petalIndex)
        )
        vertexIndex += 1
        
        cellVertices[vertexIndex] = VertexWithSeed(
          position: finalVertex2,
          color: SIMD3<Float>(1.0, 1.0, 1.0),
          seed: Int32(petalIndex)
        )
        vertexIndex += 1
        
        cellVertices[vertexIndex] = VertexWithSeed(
          position: finalVertex3,
          color: SIMD3<Float>(1.0, 1.0, 1.0),
          seed: Int32(petalIndex)
        )
        vertexIndex += 1
      }
    }
  }
  
    // 在花朵的面上生成随机位置
  private func generateRandomPositionInFlower() -> SIMD3<Float> {
      // 花朵有8个面，随机选择一个面
    let faceIndex = Int.random(in: 0..<8)
    
      // 生成面上的随机重心坐标
    let u = Float.random(in: 0...1)
    let v = Float.random(in: 0...(1 - u))
    let w = 1 - u - v
    
      // 花朵的8个面的顶点（每个面是三角形）
    let faces: [[SIMD3<Float>]] = [
      // 上半部分的4个面
      [
        SIMD3<Float>(flowerRadius, 0, 0), SIMD3<Float>(0, flowerRadius, 0),
        SIMD3<Float>(0, 0, flowerRadius),
      ],
      [
        SIMD3<Float>(0, flowerRadius, 0), SIMD3<Float>(-flowerRadius, 0, 0),
        SIMD3<Float>(0, 0, flowerRadius),
      ],
      [
        SIMD3<Float>(-flowerRadius, 0, 0), SIMD3<Float>(0, flowerRadius, 0),
        SIMD3<Float>(0, 0, -flowerRadius),
      ],
      [
        SIMD3<Float>(0, flowerRadius, 0), SIMD3<Float>(flowerRadius, 0, 0),
        SIMD3<Float>(0, 0, -flowerRadius),
      ],
      // 下半部分的4个面
      [
        SIMD3<Float>(flowerRadius, 0, 0), SIMD3<Float>(0, 0, flowerRadius),
        SIMD3<Float>(0, -flowerRadius, 0),
      ],
      [
        SIMD3<Float>(0, 0, flowerRadius), SIMD3<Float>(-flowerRadius, 0, 0),
        SIMD3<Float>(0, -flowerRadius, 0),
      ],
      [
        SIMD3<Float>(-flowerRadius, 0, 0), SIMD3<Float>(0, 0, -flowerRadius),
        SIMD3<Float>(0, -flowerRadius, 0),
      ],
      [
        SIMD3<Float>(0, 0, -flowerRadius), SIMD3<Float>(flowerRadius, 0, 0),
        SIMD3<Float>(0, -flowerRadius, 0),
      ],
    ]
    
    let selectedFace = faces[faceIndex]
    
      // 使用重心坐标在三角形面上生成随机点
    let randomPoint = u * selectedFace[0] + v * selectedFace[1] + w * selectedFace[2]
    
    return randomPoint
  }
  
    // 检查点是否在花朵内
  private func isPointInFlower(point: SIMD3<Float>) -> Bool {
    let x = abs(point.x)
    let y = abs(point.y)
    let z = abs(point.z)
    
      // 花朵的约束条件：|x| + |y| + |z| <= radius
    return (x + y + z) <= flowerRadius
  }
  
    // 辅助函数：创建3D旋转矩阵
  private func createRotationMatrix(x: Float, y: Float, z: Float) -> simd_float3x3 {
    let cosX = cos(x)
    let sinX = sin(x)
    let cosY = cos(y)
    let sinY = sin(y)
    let cosZ = cos(z)
    let sinZ = sin(z)
    
      // 组合旋转矩阵 (Z * Y * X)
    return simd_float3x3(
      SIMD3<Float>(cosY * cosZ, -cosY * sinZ, sinY),
      SIMD3<Float>(
        sinX * sinY * cosZ + cosX * sinZ, -sinX * sinY * sinZ + cosX * cosZ, -sinX * cosY),
      SIMD3<Float>(-cosX * sinY * cosZ + sinX * sinZ, cosX * sinY * sinZ + sinX * cosZ, cosX * cosY)
    )
  }
  
    // 移除compute相关函数
  
  private func createFlowerIndexBuffer(device: MTLDevice) {
    let bufferLength = MemoryLayout<UInt32>.stride * indexesCount
    indexBuffer = device.makeBuffer(length: bufferLength)!
    indexBuffer.label = "Flower petal index buffer"
    
    let cellIndices = indexBuffer.contents().bindMemory(
      to: UInt32.self, capacity: indexesCount)
    
      // 为每个花瓣生成索引（每个花瓣3个顶点，按顺序排列）
    for petalIndex in 0..<totalPetals {
      let baseVertexIndex = petalIndex * verticesPerPetal
      let baseIndexIndex = petalIndex * indexesPerPetal
      
        // 花瓣的三个顶点索引
      cellIndices[baseIndexIndex] = UInt32(baseVertexIndex)
      cellIndices[baseIndexIndex + 1] = UInt32(baseVertexIndex + 1)
      cellIndices[baseIndexIndex + 2] = UInt32(baseVertexIndex + 2)
    }
  }
  
  private func createPetalDataBuffer(device: MTLDevice) {
    let bufferSize = MemoryLayout<CellBase>.stride * totalPetals
    petalDataBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared)!
    
    let petalBase = petalDataBuffer.contents().bindMemory(
      to: CellBase.self, capacity: totalPetals)
    
      // 为每朵花生成位置
    let flowerCenters: [SIMD3<Float>] = [
      SIMD3<Float>(-0.8, 0.0, -2.5),  // 左侧花朵
      SIMD3<Float>(0.8, 0.0, -2.5),  // 右侧花朵
    ]
    
    var petalIndex = 0
    
    for flowerId in 0..<flowerCount {
      let flowerCenter = flowerCenters[flowerId]
      let flowerColor: SIMD3<Float> =
      flowerId == 0 ? SIMD3<Float>(1.0, 0.2, 0.2) : SIMD3<Float>(0.2, 1.0, 0.2)
      
      for _ in 0..<petalPerFlower {
        let relativePosition = generateRandomPositionInFlower()
        let petalSize = Float.random(in: petalMinSize...petalMaxSize)
        
        petalBase[petalIndex] = CellBase(
          position: relativePosition,
          color: flowerColor,
          flowerId: Float(flowerId),
          flowerCenter: flowerCenter,
          rotationAngle: 0.0,
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
    
    var params_data = Params(
      viewerPosition: gestureManager.viewerPosition,
      time: getTimeSinceStart(),
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
      petalDataBuffer, offset: 0, index: BufferIndex.base.rawValue)
    
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
  
    // 协议要求的方法，但由于已移除compute shader，这些方法为空实现
  func resetComputeState() {
      // 不再需要compute state重置
  }
  
  func computeCommandCommit() {
      // 不再需要compute command提交
  }
}
