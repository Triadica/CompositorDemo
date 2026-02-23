/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
The app's main renderer.
*/

import CompositorServices
import Metal
import MetalKit
import Spatial
import SwiftUI
import simd

/// defines the custom renderer protocol
@MainActor
protocol CustomRenderer {
  func drawCommand(frame: LayerRenderer.Frame) throws -> TintDrawCommand
  func encodeDraw(
    _ drawCommand: TintDrawCommand, encoder: MTLRenderCommandEncoder,
    drawable: LayerRenderer.Drawable,
    device: MTLDevice, tintValue: Float, buffer: MTLBuffer, indexBuffer: MTLBuffer
  )
  func updateUniformBuffers(_ drawCommand: TintDrawCommand, drawable: LayerRenderer.Drawable)
    async

  /// Reset the compute state
  func resetComputeState()

  func computeCommandCommit()

  var vertexBuffer: MTLBuffer! { get set }
  var indexBuffer: MTLBuffer! { get set }

  /// handle spatial events
  func onSpatialEvents(events: SpatialEventCollection)
}

/// Represents a ping-pong or bilateral oscillation behavior
///
/// The ping-pong pattern describes a value that moves back and forth between two points,
/// similar to how a ping-pong ball bounces between players.
class PingPongBuffer {
  var currentBuffer: MTLBuffer
  var nextBuffer: MTLBuffer

  /// Creates a new pair of ping-pong buffer.
  init(device: MTLDevice, length: Int) {
    guard let safeBuffer = device.makeBuffer(length: length, options: .storageModeShared),
      let safeBufferB = device.makeBuffer(length: length, options: .storageModeShared)
    else {
      fatalError("Failed to create ping-pong buffer")
    }
    currentBuffer = safeBuffer
    nextBuffer = safeBufferB
  }

  /// Swaps the current and next buffers.
  func swap() {
    (currentBuffer, nextBuffer) = (nextBuffer, currentBuffer)
  }

  /// add label
  func addLabel(_ label: String) {
    currentBuffer.label = label
    nextBuffer.label = label
  }

  func copyToNext() {
    nextBuffer.contents().copyMemory(
      from: currentBuffer.contents(), byteCount: currentBuffer.length)
  }
}

@RendererActor
struct TintDrawCommand {
  @RendererActor
  fileprivate struct DrawCommand {
    let buffer: MTLBuffer
    let vertexCount: Int
  }

  fileprivate let drawCommand: DrawCommand
  let frameIndex: LayerFrameIndex
  let uniforms: MTLBuffer & Sendable

  @MainActor
  init(frameIndex: LayerFrameIndex, uniforms: MTLBuffer, verticesCount: Int) {
    self.drawCommand = DrawCommand(buffer: uniforms, vertexCount: verticesCount)  // not really used
    self.frameIndex = frameIndex
    self.uniforms = uniforms
  }
}

extension MemoryLayout {
  static var uniformStride: Int {
    // The 256 byte aligned size of the uniform structure.
    (size + 0xFF) & -0x100
  }
}

extension LayerRenderer.Clock.Instant.Duration {
  var timeInterval: TimeInterval {
    let nanoseconds = TimeInterval(components.attoseconds / 1_000_000_000)
    return TimeInterval(components.seconds) + (nanoseconds / TimeInterval(NSEC_PER_SEC))
  }
}

extension MTLDevice {
  var supportsMSAA: Bool {
    supportsTextureSampleCount(4) && supports32BitMSAA
  }
}

@globalActor actor RendererActor {
  static var shared = RendererActor()
}

@RendererActor
class Renderer {
  
  private let appModel: AppModel

  
  private let customRenderer: CustomRenderer

  
  private let device: MTLDevice
  private let supportsMSAA: Bool
  private let commandQueue: MTLCommandQueue
  nonisolated static let maxFramesInFlight: UInt64 = 3
  private let depthState: MTLDepthStencilState
  private let layerRenderer: LayerRenderer
  private var multisampleRenderTargets: [(color: MTLTexture, depth: MTLTexture)?]
  private var frameCounter: Int = 0
  private var lastWorldTrackingState: DataProviderState = .stopped
  private var worldTrackingStateChangeCount: Int = 0

  
  private let arSession: ARKitSession
  private let worldTracking: WorldTrackingProvider

  init(
    _ layerRenderer: LayerRenderer,
    _ appModel: AppModel,
    _ customRenderer: CustomRenderer
  ) throws {
    self.appModel = appModel

    self.customRenderer = customRenderer

    self.layerRenderer = layerRenderer
    self.device = layerRenderer.device
    supportsMSAA = layerRenderer.device.supportsMSAA
    self.commandQueue = self.device.makeCommandQueue()!
    multisampleRenderTargets = .init(repeating: nil, count: Int(Self.maxFramesInFlight))

    let depthStateDescriptor = MTLDepthStencilDescriptor()
    depthStateDescriptor.depthCompareFunction = MTLCompareFunction.greater
    depthStateDescriptor.isDepthWriteEnabled = true
    self.depthState = device.makeDepthStencilState(descriptor: depthStateDescriptor)!

    arSession = ARKitSession()
    worldTracking = WorldTrackingProvider()
  }

  nonisolated static func defaultRenderPipelineDescriptor(layerRenderer: LayerRenderer)
    -> MTLRenderPipelineDescriptor
  {
    let pipelineDescriptor = MTLRenderPipelineDescriptor()

    pipelineDescriptor.rasterSampleCount = layerRenderer.device.supportsMSAA ? 4 : 1
    pipelineDescriptor.colorAttachments[0].pixelFormat = layerRenderer.configuration.colorFormat
    pipelineDescriptor.depthAttachmentPixelFormat = layerRenderer.configuration.depthFormat
    pipelineDescriptor.maxVertexAmplificationCount = layerRenderer.properties.viewCount

    return pipelineDescriptor
  }

  private func memorylessTexture(from texture: MTLTexture) -> MTLTexture {
    let descriptor = MTLTextureDescriptor.texture2DDescriptor(
      pixelFormat: texture.pixelFormat,
      width: texture.width,
      height: texture.height,
      mipmapped: false)

    if supportsMSAA {
      descriptor.textureType =
        texture.textureType == .type2DArray ? .type2DMultisampleArray : .type2DMultisample
    } else {
      descriptor.textureType = texture.textureType
    }
    descriptor.sampleCount = supportsMSAA ? 4 : 1
    descriptor.storageMode = .memoryless
    descriptor.arrayLength = texture.arrayLength
    descriptor.usage = texture.usage

    return texture.device.makeTexture(descriptor: descriptor)!
  }

  private func getMultisampleRenderTarget(
    for frame: LayerRenderer.Frame,
    drawable: LayerRenderer.Drawable
  ) -> (color: MTLTexture, depth: MTLTexture) {

    let intermediateRenderTargetIndex = Int(frame.frameIndex % Self.maxFramesInFlight)

    if let (color, depth) = multisampleRenderTargets[intermediateRenderTargetIndex],
      color.width == drawable.colorTextures[0].width,
      color.height == drawable.colorTextures[0].height
    {
      return (color, depth)
    } else {
      let color = memorylessTexture(from: drawable.colorTextures[0])
      let depth = memorylessTexture(from: drawable.depthTextures[0])
      multisampleRenderTargets[intermediateRenderTargetIndex] = (color, depth)
      return (color, depth)
    }
  }
}

extension Renderer {
  func renderLoop() async throws {
    let timestamp = Date().formatted(.dateTime.minute().second())
    print("[\(timestamp)] Renderer: Starting render loop")
    
    // Setup ARKit Session
    let authorizations: [ARKitSession.AuthorizationType] = WorldTrackingProvider
      .requiredAuthorizations
    let dataProviders: [any DataProvider] = [worldTracking]

    print("[\(timestamp)] Renderer: Requesting ARKit authorization")
    _ = await arSession.requestAuthorization(for: authorizations)
    
    print("[\(timestamp)] Renderer: Starting ARKit session")
    try await arSession.run(dataProviders)
    
    // 等待ARKit会话完全启动
    var retryCount = 0
    while worldTracking.state != .running && retryCount < 30 {
      try await Task.sleep(nanoseconds: 100_000_000) // 100ms
      retryCount += 1
      print("[\(Date().formatted(.dateTime.minute().second()))] Renderer: Waiting for ARKit, current state: \(worldTracking.state), retry: \(retryCount)")
    }
    
    if worldTracking.state != .running {
      print("[\(Date().formatted(.dateTime.minute().second()))] Renderer: Warning - ARKit world tracking failed to start after 3 seconds, current state: \(worldTracking.state)")
    } else {
      print("[\(timestamp)] Renderer: ARKit world tracking started successfully")
    }
    
    var frameCounter = 0
    let startTime = Date()
    
    // Render loop
    while true {
      frameCounter += 1
      
      if layerRenderer.state == .invalidated {
        let endTime = Date()
        let duration = endTime.timeIntervalSince(startTime)
        print("[\(Date().formatted(.dateTime.minute().second()))] Renderer: Layer is invalidated after \(frameCounter) frames (\(String(format: "%.2f", duration))s)")
        Task { @MainActor in
          arSession.stop()
        }
        return
      } else if layerRenderer.state == .paused {
        print("[\(Date().formatted(.dateTime.minute().second()))] Renderer: Layer is paused, waiting...")
        layerRenderer.waitUntilRunning()
        print("[\(Date().formatted(.dateTime.minute().second()))] Renderer: Layer resumed")
        continue
      } else {
        do {
          // 监控ARKit状态变化
          if worldTracking.state != lastWorldTrackingState {
            worldTrackingStateChangeCount += 1
            print("[\(Date().formatted(.dateTime.minute().second()))] Renderer: ARKit state changed from \(lastWorldTrackingState) to \(worldTracking.state) (change #\(worldTrackingStateChangeCount))")
            lastWorldTrackingState = worldTracking.state
            
            // 如果从暂停状态恢复到运行状态，记录恢复信息
            if worldTracking.state == .running {
              print("[\(Date().formatted(.dateTime.minute().second()))] Renderer: ARKit world tracking recovered successfully")
            }
          }
          
          await customRenderer.computeCommandCommit()
          try await self.renderFrame()
          
          // 每1000帧输出一次状态
          if frameCounter % 1000 == 0 {
            let currentTime = Date()
            let duration = currentTime.timeIntervalSince(startTime)
            print("[\(currentTime.formatted(.dateTime.minute().second()))] Renderer: Completed \(frameCounter) frames in \(String(format: "%.2f", duration))s, ARKit state: \(worldTracking.state)")
          }
        } catch {
          print("[\(Date().formatted(.dateTime.minute().second()))] Render frame error: \(error)")
          // 继续渲染循环，不因单帧错误而退出
          continue
        }
      }
    }
  }

  private func renderPassDescriptor(
    _ frame: LayerRenderer.Frame, _ drawable: LayerRenderer.Drawable
  ) -> MTLRenderPassDescriptor {
    // Create the render descriptor with the drawable targets.
    let renderPassDescriptor = MTLRenderPassDescriptor()

    if supportsMSAA {
      let (colorTarget, depthTarget) = getMultisampleRenderTarget(
        for: frame, drawable: drawable)
      renderPassDescriptor.colorAttachments[0].texture = colorTarget
      renderPassDescriptor.colorAttachments[0].resolveTexture = drawable.colorTextures[0]
      renderPassDescriptor.colorAttachments[0].storeAction = .multisampleResolve

      renderPassDescriptor.depthAttachment.texture = depthTarget
      renderPassDescriptor.depthAttachment.resolveTexture = drawable.depthTextures[0]
      renderPassDescriptor.depthAttachment.storeAction = .multisampleResolve
    } else {
      renderPassDescriptor.colorAttachments[0].texture = drawable.colorTextures[0]
      renderPassDescriptor.colorAttachments[0].storeAction = .store

      renderPassDescriptor.depthAttachment.texture = drawable.depthTextures[0]
      renderPassDescriptor.depthAttachment.storeAction = .store
    }

    renderPassDescriptor.colorAttachments[0].loadAction = .clear
    renderPassDescriptor.colorAttachments[0].clearColor = .init(
      red: 0.0, green: 0.0, blue: 0.0, alpha: 0.0)
    renderPassDescriptor.depthAttachment.loadAction = .clear
    renderPassDescriptor.depthAttachment.clearDepth = 0.0

    renderPassDescriptor.rasterizationRateMap = drawable.rasterizationRateMaps.first
    if layerRenderer.configuration.layout == .layered {
      renderPassDescriptor.renderTargetArrayLength = drawable.views.count
    }

    return renderPassDescriptor
  }

  func renderFrame() async throws {
    guard let frame = layerRenderer.queryNextFrame() else { 
      print("[\(Date().formatted(.dateTime.minute().second()))] Renderer: Failed to query next frame")
      return 
    }

    frame.startUpdate()

    guard let timing = frame.predictTiming() else { 
      print("[\(Date().formatted(.dateTime.minute().second()))] Renderer: Failed to predict timing")
      return 
    }

    // Update scene and generate draw commands.
    let lampsDrawCommand = try await Task { @MainActor in
      return try await customRenderer.drawCommand(frame: frame)
    }.result.get()

    // Query the drawable after scene update to avoid blocking on the drawable.
    guard let drawable = frame.queryDrawable() else { 
      print("[\(Date().formatted(.dateTime.minute().second()))] Renderer: Failed to query drawable")
      return 
    }

    guard let commandBuffer = commandQueue.makeCommandBuffer() else {
      print("[\(Date().formatted(.dateTime.minute().second()))] Renderer: Failed to create command buffer")
      fatalError("Failed to create command buffer")
    }

    // Encode the GPU render commands.
    let renderPassDescriptor = self.renderPassDescriptor(frame, drawable)

    guard
      let renderEncoder = commandBuffer.makeRenderCommandEncoder(
        descriptor: renderPassDescriptor)
    else { fatalError("Failed to create render encoder") }

    renderEncoder.label = "Primary Render Encoder"
    renderEncoder.pushDebugGroup("Mesh drawing")
    renderEncoder.setFrontFacing(.counterClockwise)
    renderEncoder.setDepthStencilState(depthState)

    let viewports = drawable.views.map { $0.textureMap.viewport }

    renderEncoder.setViewports(viewports)

    if drawable.views.count > 1 {
      var viewMappings = (0..<drawable.views.count).map {
        MTLVertexAmplificationViewMapping(
          viewportArrayIndexOffset: UInt32($0), renderTargetArrayIndexOffset: UInt32($0))
      }
      renderEncoder.setVertexAmplificationCount(viewports.count, viewMappings: &viewMappings)
    }
    await customRenderer.encodeDraw(
      lampsDrawCommand, encoder: renderEncoder, drawable: drawable, device: device,
      tintValue: appModel.opacity,
      buffer: customRenderer.vertexBuffer,
      indexBuffer: customRenderer.indexBuffer
    )

    renderEncoder.popDebugGroup()
    renderEncoder.endEncoding()

    frame.endUpdate()

    // Pace frames by waiting for the optimal prediction time.
    try await LayerRenderer.Clock().sleep(until: timing.optimalInputTime, tolerance: nil)

    // Start submitting the updated frame.
    frame.startSubmission()

    // Get the drawable device anchor state at presentation time.
    // 检查世界跟踪提供者状态，避免在未运行时查询设备锚点
    if worldTracking.state == .running {
      let time = LayerRenderer.Clock.Instant.epoch.duration(
        to: drawable.frameTiming.presentationTime
      ).timeInterval
      let deviceAnchor = worldTracking.queryDeviceAnchor(atTimestamp: time)
      drawable.deviceAnchor = deviceAnchor
    } else {
      // 如果世界跟踪未运行，记录状态但不频繁打印
       if frameCounter % 60 == 0 { // 每60帧（约1秒）打印一次
         print("[\(Date().formatted(.dateTime.minute().second()))] Renderer: World tracking not running (state: \(worldTracking.state)), skipping device anchor")
       }
      drawable.deviceAnchor = nil
    }

    // Update the renderer uniforms using the latest device anchor.
    await customRenderer.updateUniformBuffers(lampsDrawCommand, drawable: drawable)

    drawable.encodePresent(commandBuffer: commandBuffer)

    // 添加命令缓冲区完成处理程序
    commandBuffer.addCompletedHandler { buffer in
      if let error = buffer.error {
        print("[\(Date().formatted(.dateTime.minute().second()))] Renderer: Command buffer error: \(error)")
      }
    }
    
    commandBuffer.commit()

    frame.endSubmission()
  }
}
