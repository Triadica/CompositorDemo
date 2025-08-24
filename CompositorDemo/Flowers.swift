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
// 性能优化：使用编译时常量减少运行时计算
private let flowerCount: Int = 3  // 3朵花
private let petalsPerFlower: Int = 6  // 每朵花6个花瓣
private let segmentsPerPetal: Int = 20  // 每个花瓣的曲线段数
private let linesPerPetal: Int = 15  // 每个花瓣内部填充线条数

// 预计算的常量，避免运行时重复计算
private let totalPetals = 18  // flowerCount * petalsPerFlower = 3 * 6
private let totalOutlineSegments = 360  // totalPetals * segmentsPerPetal = 18 * 20
private let totalFillLines = 270  // totalPetals * linesPerPetal = 18 * 15
private let totalRenderElements = 630  // totalOutlineSegments + totalFillLines = 360 + 270
private let verticesCount = 1260  // totalRenderElements * 2
private let indexesCount = 1260  // totalRenderElements * 2

// 花朵尺寸常量
private let flowerSize: Float = 0.3  // 花朵直径约0.3米
private let flowerRadius: Float = 0.15  // flowerSize * 0.5
private let petalLength: Float = 0.08  // 花瓣长度
private let petalWidth: Float = 0.04   // 花瓣宽度

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
  
  // 性能优化：预分配参数缓冲区，避免每帧创建
  private var paramsBuffer: MTLBuffer!
  
  var gestureManager: GestureManager = GestureManager(onScene: false)
  
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
    
    var vertexIndex = 0
    var elementIndex = 0
    
    // 为每朵花生成花瓣
    for flowerId in 0..<flowerCount {
      for petalId in 0..<petalsPerFlower {
        let petalAngle = Float(petalId) * 2.0 * Float.pi / Float(petalsPerFlower)
        
        // 生成花瓣轮廓曲线
        for segmentId in 0..<segmentsPerPetal {
          let t1 = Float(segmentId) / Float(segmentsPerPetal)
          let t2 = Float(segmentId + 1) / Float(segmentsPerPetal)
          
          // 使用心形曲线变形创建花瓣形状
          let point1 = createPetalPoint(t: t1, petalAngle: petalAngle)
          let point2 = createPetalPoint(t: t2, petalAngle: petalAngle)
          
          // 创建轮廓线段的两个顶点
          cellVertices[vertexIndex] = VertexWithSeed(
            position: point1,
            color: SIMD3<Float>(1.0, 0.8, 0.9), // 粉色轮廓
            seed: Int32(flowerId * petalsPerFlower + petalId)
          )
          vertexIndex += 1
          
          cellVertices[vertexIndex] = VertexWithSeed(
            position: point2,
            color: SIMD3<Float>(1.0, 0.8, 0.9),
            seed: Int32(flowerId * petalsPerFlower + petalId)
          )
          vertexIndex += 1
          elementIndex += 1
        }
        
        // 生成花瓣内部填充线条 - 从中心向边缘的简单线条
        for lineId in 0..<linesPerPetal {
          let t = Float(lineId) / Float(linesPerPetal - 1)
          
          // 从花瓣中心向边缘绘制线条
          let centerPoint = SIMD3<Float>(0, 0, 0)
          
          // 在花瓣边缘上找到对应的点
          let edgePoint = createPetalPoint(t: t, petalAngle: petalAngle)
          
          // 颜色渐变：中心深，边缘浅
          let fillColor = SIMD3<Float>(1.0, 0.7 + 0.2 * t, 0.8 + 0.1 * t)
          
          cellVertices[vertexIndex] = VertexWithSeed(
            position: centerPoint,
            color: fillColor,
            seed: Int32(flowerId * petalsPerFlower + petalId)
          )
          vertexIndex += 1
          
          cellVertices[vertexIndex] = VertexWithSeed(
            position: edgePoint,
            color: fillColor,
            seed: Int32(flowerId * petalsPerFlower + petalId)
          )
          vertexIndex += 1
          elementIndex += 1
        }
       }
     }
   }
   
   /// 创建花瓣上的点，使用简单的椭圆形状
  private func createPetalPoint(t: Float, petalAngle: Float) -> SIMD3<Float> {
    // t 从 0 到 1，描述花瓣轮廓
    // 使用椭圆方程创建花瓣形状
    
    let angle = t * 2.0 * Float.pi
    
    // 创建椭圆形花瓣，长轴是短轴的2倍
    let x = petalLength * cos(angle)
    let y = petalWidth * sin(angle)
    
    // 应用花瓣在花朵中的旋转
    let cosPetalAngle = cos(petalAngle)
    let sinPetalAngle = sin(petalAngle)
    let rotatedX = x * cosPetalAngle - y * sinPetalAngle
    let rotatedY = x * sinPetalAngle + y * cosPetalAngle
    
    return SIMD3<Float>(rotatedX, rotatedY, 0)
  }


  


  
    
  
  private func createFlowerIndexBuffer(device: MTLDevice) {
    let bufferLength = MemoryLayout<UInt32>.stride * indexesCount
    indexBuffer = device.makeBuffer(length: bufferLength)!
    indexBuffer.label = "Flower petal index buffer"
    
    let cellIndices = indexBuffer.contents().bindMemory(
      to: UInt32.self, capacity: indexesCount)
    
    var indexOffset = 0
    
    // 为每朵花的每个花瓣生成线段索引
    for flowerId in 0..<flowerCount {
      for petalId in 0..<petalsPerFlower {
        let baseVertexIndex = (flowerId * petalsPerFlower + petalId) * (segmentsPerPetal + linesPerPetal) * 2
        
        // 花瓣轮廓线段索引
        for segmentId in 0..<segmentsPerPetal {
          let segmentBaseIndex = baseVertexIndex + segmentId * 2
          cellIndices[indexOffset] = UInt32(segmentBaseIndex)
          cellIndices[indexOffset + 1] = UInt32(segmentBaseIndex + 1)
          indexOffset += 2
        }
        
        // 花瓣填充线段索引
        let fillBaseIndex = baseVertexIndex + segmentsPerPetal * 2
        for lineId in 0..<linesPerPetal {
          let lineBaseIndex = fillBaseIndex + lineId * 2
          cellIndices[indexOffset] = UInt32(lineBaseIndex)
          cellIndices[indexOffset + 1] = UInt32(lineBaseIndex + 1)
          indexOffset += 2
        }
      }
    }
  }
  
  private func createPetalDataBuffer(device: MTLDevice) {
    let bufferSize = MemoryLayout<CellBase>.stride * totalPetals
    petalDataBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared)!
    
    let petalBase = petalDataBuffer.contents().bindMemory(
      to: CellBase.self, capacity: totalPetals)
    
    // 改进的花朵分布 - 创建更自然的3D空间分布
    let flowerCenters: [SIMD3<Float>] = [
      SIMD3<Float>(-1.2, 0.3, -2.8),   // 左上花朵
      SIMD3<Float>(1.0, -0.2, -2.2),   // 右下花朵
      SIMD3<Float>(0.0, 0.8, -3.5)     // 中央后方花朵
    ]
    
    // 为每朵花设置不同的颜色主题
    let flowerColors: [SIMD3<Float>] = [
      SIMD3<Float>(1.0, 0.3, 0.4),     // 粉红色
      SIMD3<Float>(0.9, 0.7, 0.2),     // 金黄色
      SIMD3<Float>(0.7, 0.4, 0.9)      // 紫色
    ]
    
    var petalIndex = 0
    
    for flowerId in 0..<flowerCount {
      let flowerCenter = flowerCenters[flowerId]
      let flowerColor = flowerColors[flowerId]
      
      // 为每个花瓣设置不同的角度和属性
      for petalId in 0..<petalsPerFlower {
        let petalAngle = Float(petalId) * (2.0 * Float.pi / Float(petalsPerFlower))
        let relativePosition = SIMD3<Float>(0, 0, 0)  // 花瓣相对位置将在着色器中计算
        let petalSize = Float.random(in: petalWidth...petalLength)
        
        // 添加轻微的颜色变化，使每个花瓣略有不同
        let colorVariation = Float.random(in: 0.9...1.1)
        let variedColor = flowerColor * colorVariation
        
        petalBase[petalIndex] = CellBase(
          position: relativePosition,
          color: variedColor,
          flowerId: Float(flowerId),
          flowerCenter: flowerCenter,
          petalId: Float(petalId),
          lineType: 0.0,  // 轮廓线
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
      type: .line,
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
