/*
 See the LICENSE.txt file for this sample's licensing information.

 Abstract:
 A renderer that displays a set of color swatches.
 */

import CompositorServices
import Metal
import MetalKit
import Spatial
import SwiftUI
import simd

private let maxFramesInFlight = 3

/// how many lines for this attractor
private let linesCount: Int = 80000
/// how many rectangles in a line
private let lineGroupSize: Int = 2
/// 1 for leading point, others are following points
private var controlCountPerLine: Int {
  lineGroupSize + 1
}
/// all control points in the scene
private var controlCount: Int {
  linesCount * controlCountPerLine
}

private let verticesCount = controlCount * 6

/// rectangle indexes per rectangle
private let indexesCount: Int = controlCount * 6

private struct SpreadInBallBase {
  var position: SIMD3<Float>
  var color: SIMD3<Float>
  var velocity: SIMD3<Float> = SIMD3<Float>(0, 0, 0)
}

private struct SpreadInBallParams {
  var time: Float
  var groupSize: Int32 = Int32(lineGroupSize)
  var viewerPosition: SIMD3<Float>
  var viewerScale: Float
  var viewerRotation: Float = 0.0
  var _padding: SIMD2<Float> = SIMD2<Float>(0, 0)  // Pad to 48 bytes, remove if shader expects 36 bytes
}

@MainActor
class SpreadInBallRenderer: CustomRenderer {
  private let renderPipelineState: MTLRenderPipelineState & Sendable

  private var uniformsBuffer: [MTLBuffer]
  /// a buffer to hold the vertices of the lamp
  var vertexBuffer: MTLBuffer!

  var indexBuffer: MTLBuffer!

  let computeDevice: MTLDevice
  var computeBuffer: PingPongBuffer?
  let computePipeLine: MTLComputePipelineState
  let computeCommandQueue: MTLCommandQueue

  var gestureManager: GestureManager = GestureManager()
  
  // 优化：预分配参数缓冲区，避免每帧创建新buffer
  private var paramsBuffers: [MTLBuffer]
  private var currentParamsBufferIndex = 0
  
  // 性能监控
  private var lastFrameTime: CFTimeInterval = 0
  private var frameCount = 0
  private var averageFrameTime: Double = 0

  init(layerRenderer: LayerRenderer) throws {
    print("🚀 SpreadInBall: 开始初始化渲染器")
    
    uniformsBuffer = (0..<Renderer.maxFramesInFlight).map { _ in
      layerRenderer.device.makeBuffer(length: MemoryLayout<PathProperties>.uniformStride)!
    }
    print("✅ SpreadInBall: uniforms缓冲区创建完成")
    
    // 初始化参数缓冲区池
    paramsBuffers = (0..<maxFramesInFlight).map { _ in
      layerRenderer.device.makeBuffer(length: MemoryLayout<SpreadInBallParams>.stride, options: .storageModeShared)!
    }
    print("✅ SpreadInBall: 参数缓冲区池创建完成")

    renderPipelineState = try Self.makeRenderPipelineDescriptor(layerRenderer: layerRenderer)
    print("✅ SpreadInBall: 渲染管线状态创建完成")

    self.computeDevice = MTLCreateSystemDefaultDevice()!
    let library = computeDevice.makeDefaultLibrary()!
    let attractorUpdateBase = library.makeFunction(name: "spreadInBallComputeShader")!
    computePipeLine = try computeDevice.makeComputePipelineState(function: attractorUpdateBase)
    print("✅ SpreadInBall: 计算管线创建完成")

    computeCommandQueue = computeDevice.makeCommandQueue()!

    self.createAttractorVerticesBuffer(device: layerRenderer.device)
    self.createAttractorIndexBuffer(device: layerRenderer.device)
    self.createAttractorComputeBuffer(device: layerRenderer.device)
    
    print("🎉 SpreadInBall: 渲染器初始化完成，控制点数量: \(controlCount)，顶点数量: \(verticesCount)")
  }

  /// create and sets the vertices of the lamp
  private func createAttractorVerticesBuffer(device: MTLDevice) {
    let bufferLength = MemoryLayout<AttractorCellVertex>.stride * verticesCount
    vertexBuffer = device.makeBuffer(length: bufferLength)!
    vertexBuffer.label = "Attractor vertex buffer"
    var attractorVertices: UnsafeMutablePointer<AttractorCellVertex> {
      vertexBuffer.contents().assumingMemoryBound(to: AttractorCellVertex.self)
    }

    for i in 0..<linesCount {
      let baseIndex = i * lineGroupSize * 6

      for j in 0..<lineGroupSize {
        let index = baseIndex + j * 6
        // set 6 vertices for each cell
        // 1st vertex
        attractorVertices[index] = AttractorCellVertex(
          position: SIMD3<Float>(0, 0, 0),
          lineNumber: Int32(i),
          groupNumber: Int32(j),
          cellSide: 0)
        // 2nd vertex
        attractorVertices[index + 1] = AttractorCellVertex(
          position: SIMD3<Float>(0, 0, 0),
          lineNumber: Int32(i),
          groupNumber: Int32(j),
          cellSide: 1)
        // 3rd vertex
        attractorVertices[index + 2] = AttractorCellVertex(
          position: SIMD3<Float>(0, 0, 0),
          lineNumber: Int32(i),
          groupNumber: Int32(j),
          cellSide: 2)
        // 4th vertex
        attractorVertices[index + 3] = AttractorCellVertex(
          position: SIMD3<Float>(0, 0, 0),
          lineNumber: Int32(i),
          groupNumber: Int32(j),
          cellSide: 1)
        // 5th vertex
        attractorVertices[index + 4] = AttractorCellVertex(
          position: SIMD3<Float>(0, 0, 0),
          lineNumber: Int32(i),
          groupNumber: Int32(j),
          cellSide: 2)
        // 6th vertex
        attractorVertices[index + 5] = AttractorCellVertex(
          position: SIMD3<Float>(0, 0, 0),
          lineNumber: Int32(i),
          groupNumber: Int32(j),
          cellSide: 3)
      }

    }
  }

  func resetComputeState() {
    print("🔄 SpreadInBall: 重置计算状态")
    self.createAttractorComputeBuffer(device: computeDevice)
    print("✅ SpreadInBall: 计算状态重置完成")
  }

  private func createAttractorIndexBuffer(device: MTLDevice) {
    let bufferLength = MemoryLayout<UInt32>.stride * indexesCount
    indexBuffer = device.makeBuffer(length: bufferLength)!
    indexBuffer.label = "Lamp index buffer"

    let attractorIndices = indexBuffer.contents().bindMemory(
      to: UInt32.self, capacity: indexesCount)
    for i in 0..<indexesCount {
      attractorIndices[i] = UInt32(i)
    }

  }

  // Generate multiple small spheres positioned inside the target sphere
  private func generateSmallSpheres(
    numSpheres: Int, particlesPerSphere: Int, targetCenter: SIMD3<Float>, targetRadius: Float
  ) -> [(position: SIMD3<Float>, velocity: SIMD3<Float>, color: SIMD3<Float>)] {
    var particles: [(position: SIMD3<Float>, velocity: SIMD3<Float>, color: SIMD3<Float>)] = []

    // Create small spheres inside the target sphere (no random positioning)
    for sphereIndex in 0..<numSpheres {
      // Create two separate rings - one large, one small
      let largeRingCount = Int(Float(numSpheres) * 0.7)  // 70% of spheres in large ring
      let smallRingCount = numSpheres - largeRingCount  // 30% of spheres in small ring

      let spherePosition: SIMD3<Float>
      if sphereIndex < largeRingCount {
        // Large ring - positioned higher for greater potential energy
        let angle = Float(sphereIndex) / Float(largeRingCount) * 2.0 * Float.pi
        let distance: Float = 0.5  // Larger radius
        let height: Float = 0.3  // Higher position for greater potential energy
        spherePosition = SIMD3<Float>(
          distance * cos(angle),
          height,
          distance * sin(angle)
        )
      } else {
        // Small ring - positioned lower
        let smallIndex = sphereIndex - largeRingCount
        let angle = Float(smallIndex) / Float(smallRingCount) * 2.0 * Float.pi
        let distance: Float = 0.25  // Smaller radius
        let height: Float = -0.1  // Lower position
        spherePosition = SIMD3<Float>(
          distance * cos(angle),
          height,
          distance * sin(angle)
        )
      }
      let sphereCenter = targetCenter + spherePosition

      let smallSphereRadius = 0.05  // Fixed radius instead of random

      // Generate fixed velocity parameters for this small sphere (reduced for internal movement)
      let baseSpeed: Float
      let randomDirection = Float(sphereIndex) * 0.1  // Deterministic variation

      // Generate fixed direction for this small sphere based on its ring position
      let sphereAngle: Float
      if sphereIndex < largeRingCount {
        // Large ring: slightly slower speed and direction
        baseSpeed = Float(0.015)  // Reduced from 0.02 for slower angular velocity
        let largeIndex = sphereIndex
        sphereAngle = Float(largeIndex) / Float(largeRingCount) * 2.0 * Float.pi
      } else {
        // Small ring: slower speed and same direction pattern
        baseSpeed = Float(0.01)  // Smaller speed for small ring
        let smallIndex = sphereIndex - largeRingCount
        // Use the same angle calculation as position, but apply direction change in velocity
        sphereAngle = Float(smallIndex) / Float(smallRingCount) * 2.0 * Float.pi
      }

      let sphereDirection = SIMD3<Float>(
        -sin(sphereAngle),  // Tangent direction for circular motion
        0.0,  // Keep horizontal movement
        cos(sphereAngle)  // Tangent direction for circular motion
      )
      let normalizedDirection = normalize(sphereDirection)

      // Generate particles within this small sphere
      for particleIndex in 0..<particlesPerSphere {
        // Use global fibonacci grid function for uniform sphere distribution
        let unitPosition = fibonacciGrid(n: Float(particleIndex), total: Float(particlesPerSphere))
        let particlePosition = sphereCenter + unitPosition * Float(smallSphereRadius)

        // Calculate velocity for internal movement (no target attraction)
        let expansionDirection = normalize(particlePosition - sphereCenter)

        // Apply fixed direction for this small sphere with minimal expansion to maintain ring shape
        let ringDirection =
          sphereIndex < largeRingCount ? normalizedDirection : -normalizedDirection  // Opposite direction for small ring
        let velocity: SIMD3<Float> =
          ringDirection * baseSpeed + expansionDirection * (baseSpeed * 0.08)

        // Color based on sphere index
        let hue = Float(sphereIndex) / Float(numSpheres)
        let color = SIMD3<Float>(
          0.5 + 0.5 * cos(hue * 2.0 * Float.pi),
          0.5 + 0.5 * cos(hue * 2.0 * Float.pi + 2.0),
          0.5 + 0.5 * cos(hue * 2.0 * Float.pi + 4.0)
        )

        particles.append((position: particlePosition, velocity: velocity, color: color))
      }
    }

    return particles
  }

  private func createAttractorComputeBuffer(device: MTLDevice) {
    let bufferLength = MemoryLayout<SpreadInBallBase>.stride * controlCount

    computeBuffer = PingPongBuffer(device: device, length: bufferLength)

    guard let computeBuffer = computeBuffer else {
      print("Failed to create compute buffer")
      return
    }
    computeBuffer.addLabel("Attractor compute buffer")

    let contents = computeBuffer.currentBuffer.contents()
    let attractorBase = contents.bindMemory(to: SpreadInBallBase.self, capacity: controlCount)

    let targetCenter = SIMD3<Float>(0.0, 0.0, -1.0)
    let targetRadius: Float = 1.6

    // Generate multiple small spheres with particles
    let numSpheres: Int = 18
    let particlesPerSphere = linesCount / numSpheres
    let particles = generateSmallSpheres(
      numSpheres: numSpheres,
      particlesPerSphere: particlesPerSphere,
      targetCenter: targetCenter,
      targetRadius: targetRadius
    )

    // Fill remaining particles if any
    var particleIndex = 0
    for i in 0..<linesCount {
      let particle = particles[min(particleIndex, particles.count - 1)]
      particleIndex += 1

      for j in 0..<controlCountPerLine {
        let index = i * controlCountPerLine + j
        attractorBase[index] = SpreadInBallBase(
          position: particle.position,
          color: particle.color,
          velocity: particle.velocity
        )
      }
    }

    computeBuffer.copyToNext()
  }

  class func buildMetalVertexDescriptor() -> MTLVertexDescriptor {
    // Create a vertex descriptor specifying how Metal lays out vertices for input into the render pipeline.

    let mtlVertexDescriptor = MTLVertexDescriptor()
    var offset: Int = 0

    mtlVertexDescriptor.attributes[0].format = MTLVertexFormat.float3
    mtlVertexDescriptor.attributes[0].offset = offset
    mtlVertexDescriptor.attributes[0].bufferIndex = 0
    offset += MemoryLayout<SIMD3<Float>>.stride

    mtlVertexDescriptor.attributes[1].format = MTLVertexFormat.int
    mtlVertexDescriptor.attributes[1].offset = offset
    mtlVertexDescriptor.attributes[1].bufferIndex = 0
    offset += MemoryLayout<Int32>.stride

    mtlVertexDescriptor.attributes[2].format = MTLVertexFormat.int
    mtlVertexDescriptor.attributes[2].offset = offset
    mtlVertexDescriptor.attributes[2].bufferIndex = 0
    offset += MemoryLayout<Int32>.stride

    mtlVertexDescriptor.attributes[3].format = MTLVertexFormat.int
    mtlVertexDescriptor.attributes[3].offset = offset
    mtlVertexDescriptor.attributes[3].bufferIndex = 0
    offset += MemoryLayout<Int32>.stride

    // layout is special
    mtlVertexDescriptor.layouts[0].stride = MemoryLayout<AttractorCellVertex>.stride
    mtlVertexDescriptor.layouts[0].stepRate = 1
    mtlVertexDescriptor.layouts[0].stepFunction = MTLVertexStepFunction.perVertex

    return mtlVertexDescriptor
  }

  private static func makeRenderPipelineDescriptor(layerRenderer: LayerRenderer) throws
    -> MTLRenderPipelineState
  {
    let pipelineDescriptor = Renderer.defaultRenderPipelineDescriptor(
      layerRenderer: layerRenderer)

    let library = layerRenderer.device.makeDefaultLibrary()!

    let vertexFunction = library.makeFunction(name: "spreadInBallVertexShader")
    let fragmentFunction = library.makeFunction(name: "spreadInBallFragmentShader")

    pipelineDescriptor.fragmentFunction = fragmentFunction
    pipelineDescriptor.vertexFunction = vertexFunction

    pipelineDescriptor.label = "TriangleRenderPipeline"
    pipelineDescriptor.vertexDescriptor = self.buildMetalVertexDescriptor()

    return try layerRenderer.device.makeRenderPipelineState(descriptor: pipelineDescriptor)
  }

  func drawCommand(frame: LayerRenderer.Frame) throws -> TintDrawCommand {
    print("🎨 SpreadInBall: 创建绘制命令，帧索引: \(frame.frameIndex)")
    return TintDrawCommand(
      frameIndex: frame.frameIndex,
      uniforms: self.uniformsBuffer[Int(frame.frameIndex % Renderer.maxFramesInFlight)],
      verticesCount: verticesCount)
  }

  func computeCommandCommit() {
    let frameStartTime = CACurrentMediaTime()
    print("⚡ SpreadInBall: 开始计算命令提交，时间戳: \(frameStartTime)")
    
    guard let computeBuffer: PingPongBuffer = computeBuffer else {
      print("❌ SpreadInBall: 计算缓冲区为空，无法继续")
      return
    }
    
    guard let commandBuffer = computeCommandQueue.makeCommandBuffer() else {
      print("❌ SpreadInBall: 无法创建命令缓冲区")
      return
    }
    
    guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
      print("❌ SpreadInBall: 无法创建计算编码器")
      return
    }
    
    print("✅ SpreadInBall: 计算资源准备完成")
    
    // 添加错误监控
    commandBuffer.addCompletedHandler { [weak self] buffer in
      let completionTime = CACurrentMediaTime()
      if let error = buffer.error {
        print("❌ SpreadInBall 计算错误: \(error.localizedDescription)")
        print("❌ 错误详情: \(error)")
      } else {
        print("✅ SpreadInBall: 计算命令完成，耗时: \(String(format: "%.2f", (completionTime - frameStartTime) * 1000))ms")
      }
      
      // 性能监控
      let frameTime = CACurrentMediaTime() - frameStartTime
      self?.updatePerformanceMetrics(frameTime: frameTime)
    }

    computeEncoder.setComputePipelineState(computePipeLine)
    computeEncoder.setBuffer(computeBuffer.currentBuffer, offset: 0, index: 0)
    computeEncoder.setBuffer(computeBuffer.nextBuffer, offset: 0, index: 1)
    print("🔧 SpreadInBall: 计算管线和缓冲区设置完成")

    let delta = -Float(viewStartTime.timeIntervalSinceNow)
    let dt = delta - frameDelta
    frameDelta = delta

    var params = SpreadInBallParams(
      time: dt, viewerPosition: self.gestureManager.viewerPosition,
      viewerScale: self.gestureManager.viewerScale,
      viewerRotation: self.gestureManager.viewerRotation)
    computeEncoder.setBytes(&params, length: MemoryLayout<SpreadInBallParams>.size, index: 2)
    print("📊 SpreadInBall: 参数设置完成 - 时间: \(dt), 观察者位置: \(params.viewerPosition), 缩放: \(params.viewerScale)")
    
    // 优化线程组大小，根据设备能力动态调整
    let optimalThreadGroupSize = min(computePipeLine.maxTotalThreadsPerThreadgroup, 512)
    let threadsPerThreadgroup = MTLSize(width: optimalThreadGroupSize, height: 1, depth: 1)
    let threadGroups = MTLSize(
      width: (controlCount + optimalThreadGroupSize - 1) / optimalThreadGroupSize,
      height: 1,
      depth: 1
    )
    print("🧮 SpreadInBall: 线程组配置 - 线程组大小: \(optimalThreadGroupSize), 线程组数量: \(threadGroups.width)")
    
    computeEncoder.dispatchThreadgroups(
      threadGroups, threadsPerThreadgroup: threadsPerThreadgroup)
    computeEncoder.endEncoding()
    print("🚀 SpreadInBall: 计算任务分发完成")
    
    // 智能负载管理：在高负载时等待完成
    let shouldWait = shouldWaitForCompletion()
    print("⏱️ SpreadInBall: 负载检查 - 平均帧时间: \(String(format: "%.2f", averageFrameTime * 1000))ms, 需要等待: \(shouldWait)")
    
    if shouldWait {
      print("⏳ SpreadInBall: 高负载模式，等待命令完成")
      commandBuffer.commit()
      commandBuffer.waitUntilCompleted()
      print("✅ SpreadInBall: 命令同步完成")
    } else {
      print("🏃 SpreadInBall: 异步提交命令")
      commandBuffer.commit()
    }

    computeBuffer.swap()
    print("🔄 SpreadInBall: 缓冲区交换完成")
  }
  
  private func shouldWaitForCompletion() -> Bool {
    // 如果平均帧时间超过16.67ms（60fps），则等待完成避免积压
    return averageFrameTime > 0.0167
  }
  
  private func updatePerformanceMetrics(frameTime: Double) {
    frameCount += 1
    
    // 计算移动平均
    let alpha = 0.1 // 平滑因子
    averageFrameTime = averageFrameTime * (1 - alpha) + frameTime * alpha
    
    // 检测性能异常
    if frameTime > 0.033 { // 超过33ms（约30fps）
      print("⚠️ SpreadInBall: 检测到性能异常，帧时间: \(String(format: "%.2f", frameTime * 1000))ms")
    }
    
    // 每100帧输出一次统计
    if frameCount % 100 == 0 {
      print("📊 SpreadInBall: 性能统计 - 帧数: \(frameCount), 平均帧时间: \(String(format: "%.2f", averageFrameTime * 1000))ms")
    }
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
    print("🖼️ SpreadInBall: 开始编码绘制命令，帧索引: \(drawCommand.frameIndex)")
    
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

    var params_data = SpreadInBallParams(
      time: getTimeSinceStart(),
      viewerPosition: self.gestureManager.viewerPosition,
      viewerScale: self.gestureManager.viewerScale,
      viewerRotation: self.gestureManager.viewerRotation)

    // 使用预分配的参数缓冲区池，避免每帧创建新缓冲区
    let paramsBuffer = paramsBuffers[currentParamsBufferIndex]
    let contents = paramsBuffer.contents().bindMemory(to: SpreadInBallParams.self, capacity: 1)
    contents.pointee = params_data
    
    // 循环使用缓冲区索引
    currentParamsBufferIndex = (currentParamsBufferIndex + 1) % maxFramesInFlight

    encoder.setVertexBuffer(
      paramsBuffer,
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
    
    print("✅ SpreadInBall: 绘制命令编码完成，绘制了 \(indexesCount) 个索引")
  }

  func updateUniformBuffers(
    _ drawCommand: TintDrawCommand,
    drawable: LayerRenderer.Drawable
  ) {
    drawCommand.uniforms.contents().assumingMemoryBound(to: Uniforms.self).pointee = Uniforms(
      drawable: drawable)
  }

  /// track the position pinch started, following pinches define the velocity of moving, to update self.viewerPosition .
  /// other other chirality events are used for scaling the entity
  func onSpatialEvents(events: SpatialEventCollection) {
    print("👆 SpreadInBall: 处理空间事件，事件数量: \(events.count)")
    for event in events {
      gestureManager.onSpatialEvent(event: event)
    }
  }
}
