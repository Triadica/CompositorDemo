/*
See the LICENSE.txt file for this sample’s licensing information.

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

private let blocksCount: Int = 8

private let verticesPerBlock = 6
private let verticesCount = verticesPerBlock * blocksCount

private let indexesPerBlock = 6
private let indexesCount: Int = blocksCount * indexesPerBlock

private let blockRadius: Float = 2  // half width
private let blockHeight: Float = 20

private struct CellBase {
  var position: SIMD3<Float>
  var color: SIMD3<Float>
  var blockIdf: Float
  var velocity: SIMD3<Float> = .zero
}

private struct Params {
  /// prefer time relative to the start of the app, not delta time
  var time: Float
  var viewerPosition: SIMD3<Float>
  var viewerScale: Float
  var blocksCount: Int32 = 0
  var _padding: SIMD3<Float> = .zero  // required for 48 bytes alignment
}

@MainActor
class ImagesRenderer: CustomRenderer {
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
    let cellUpdateBase: any MTLFunction = library.makeFunction(name: "imagesComputeShader")!
    computePipeLine = try computeDevice.makeComputePipelineState(function: cellUpdateBase)

    computeCommandQueue = computeDevice.makeCommandQueue()!

    self.createBlocksVerticesBuffer(device: layerRenderer.device)
    self.createBlocksIndexBuffer(device: layerRenderer.device)
    self.createBlocksComputeBuffer(device: layerRenderer.device)
  }

  /// create and sets the vertices of the lamp
  private func createBlocksVerticesBuffer(device: MTLDevice) {
    let bufferLength = MemoryLayout<BlockVertex>.stride * verticesCount
    vertexBuffer = device.makeBuffer(length: bufferLength)!
    vertexBuffer.label = "Lamp vertex buffer"
    var cellVertices: UnsafeMutablePointer<BlockVertex> {
      vertexBuffer.contents().assumingMemoryBound(to: BlockVertex.self)
    }

    for i in 0..<blocksCount {
      let color = SIMD3<Float>(1, 1, 0)

      let baseIndex = i * 6
      let r: Float = 0.2
      let randHeight: Float = 0.1

      let p1: SIMD3<Float> = SIMD3<Float>(-r, -r, 0)
      let p2: SIMD3<Float> = SIMD3<Float>(r, -r, 0)
      let p3: SIMD3<Float> = SIMD3<Float>(r, r, 0)
      let p4: SIMD3<Float> = SIMD3<Float>(-r, r, 0)

      // front face, 126,165
      cellVertices[baseIndex] = BlockVertex(
        position: p1, color: color, seed: Int32(i), height: randHeight,
        uv: SIMD2<Float>(0, 0)
      )
      cellVertices[baseIndex + 1] = BlockVertex(
        position: p2, color: color, seed: Int32(i), height: randHeight,
        uv: SIMD2<Float>(2 * r, 0)
      )
      cellVertices[baseIndex + 2] = BlockVertex(
        position: p3, color: color, seed: Int32(i), height: randHeight,
        uv: SIMD2<Float>(2 * r, randHeight)
      )
      cellVertices[baseIndex + 3] = BlockVertex(
        position: p1, color: color, seed: Int32(i), height: randHeight,
        uv: SIMD2<Float>(0, 0)
      )
      cellVertices[baseIndex + 4] = BlockVertex(
        position: p3, color: color, seed: Int32(i), height: randHeight,
        uv: SIMD2<Float>(2 * r, randHeight)
      )
      cellVertices[baseIndex + 5] = BlockVertex(
        position: p4, color: color, seed: Int32(i), height: randHeight,
        uv: SIMD2<Float>(0, randHeight)
      )
    }

  }

  func resetComputeState() {
    self.createBlocksComputeBuffer(device: computeDevice)
  }

  private func createBlocksIndexBuffer(device: MTLDevice) {
    let bufferLength = MemoryLayout<UInt32>.stride * indexesCount
    indexBuffer = device.makeBuffer(length: bufferLength)!
    indexBuffer.label = "Images index buffer"

    let cellIndices = indexBuffer.contents().bindMemory(
      to: UInt32.self, capacity: indexesCount)
    let total = blocksCount * 6
    for i in 0..<total {
      cellIndices[i] = UInt32(i)
    }

  }

  private func createBlocksComputeBuffer(device: MTLDevice) {
    let bufferLength = MemoryLayout<CellBase>.stride * blocksCount

    computeBuffer = PingPongBuffer(device: device, length: bufferLength)

    guard let computeBuffer = computeBuffer else {
      print("Failed to create compute buffer")
      return
    }
    computeBuffer.addLabel("Lamp compute buffer")

    let contents = computeBuffer.currentBuffer.contents()
    let blocksBase = contents.bindMemory(to: CellBase.self, capacity: blocksCount)

    for i in 0..<blocksCount {

      let angle = Float(i) * (2.0 * .pi / Float(blocksCount))
      let x = cos(angle) * 2
      let z = sin(angle) * 2
      let y: Float = 1.2

      let position = SIMD3<Float>(x, y, z)
      // Random color for each lamp
      let r = Float.random(in: 0.1...1.0)
      let g = Float.random(in: 0.1...1.0)
      let b = Float.random(in: 0.1...1.0)
      let color = SIMD3<Float>(r, g, b)

      blocksBase[i] = CellBase(
        position: position, color: color, blockIdf: Float(i), velocity: .zero)

    }

    computeBuffer.copy_to_next()
  }

  class func buildMetalVertexDescriptor() -> MTLVertexDescriptor {
    // Create a vertex descriptor specifying how Metal lays out vertices for input into the render pipeline.

    let mtlVertexDescriptor = MTLVertexDescriptor()
    var offset = 0

    mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].format =
      MTLVertexFormat.float3
    mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].offset = 0
    mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].bufferIndex =
      BufferIndex.meshPositions.rawValue
    offset += MemoryLayout<SIMD3<Float>>.stride

    mtlVertexDescriptor.attributes[VertexAttribute.color.rawValue].format =
      MTLVertexFormat.float3
    mtlVertexDescriptor.attributes[VertexAttribute.color.rawValue].offset = offset
    mtlVertexDescriptor.attributes[VertexAttribute.color.rawValue].bufferIndex =
      BufferIndex.meshPositions.rawValue
    offset += MemoryLayout<SIMD3<Float>>.stride

    mtlVertexDescriptor.attributes[VertexAttribute.seed.rawValue].format = MTLVertexFormat.int
    mtlVertexDescriptor.attributes[VertexAttribute.seed.rawValue].offset = offset
    mtlVertexDescriptor.attributes[VertexAttribute.seed.rawValue].bufferIndex =
      BufferIndex.meshPositions.rawValue
    offset += MemoryLayout<Int32>.stride

    mtlVertexDescriptor.attributes[3].format = MTLVertexFormat.float
    mtlVertexDescriptor.attributes[3].offset = offset
    mtlVertexDescriptor.attributes[3].bufferIndex = 0
    offset += MemoryLayout<Float>.stride

    mtlVertexDescriptor.attributes[4].format = MTLVertexFormat.float2
    mtlVertexDescriptor.attributes[4].offset = offset
    mtlVertexDescriptor.attributes[4].bufferIndex = 0
    offset += MemoryLayout<SIMD2<Float>>.stride

    mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stride =
      MemoryLayout<BlockVertex>.stride
    mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepRate = 1
    mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepFunction =
      MTLVertexStepFunction.perVertex

    return mtlVertexDescriptor
  }

  private static func makeRenderPipelineDescriptor(layerRenderer: LayerRenderer) throws
    -> MTLRenderPipelineState
  {
    let pipelineDescriptor = Renderer.defaultRenderPipelineDescriptor(
      layerRenderer: layerRenderer)

    let library = layerRenderer.device.makeDefaultLibrary()!

    let vertexFunction = library.makeFunction(name: "imagesVertexShader")
    let fragmentFunction = library.makeFunction(name: "imagesFragmentShader")

    pipelineDescriptor.fragmentFunction = fragmentFunction
    pipelineDescriptor.vertexFunction = vertexFunction

    pipelineDescriptor.label = "TriangleRenderPipeline"
    pipelineDescriptor.vertexDescriptor = ImagesRenderer.buildMetalVertexDescriptor()

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
      time: dt, viewerPosition: gestureManager.viewerPosition,
      viewerScale: gestureManager.viewerScale,
      blocksCount: Int32(blocksCount))
    computeEncoder.setBytes(&params, length: MemoryLayout<Params>.size, index: 2)
    let threadGroupSize = min(computePipeLine.maxTotalThreadsPerThreadgroup, 256)
    let threadsPerThreadgroup = MTLSize(width: threadGroupSize, height: 1, depth: 1)
    let threadGroups = MTLSize(
      width: (blocksCount + threadGroupSize - 1) / threadGroupSize,
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

    // let bufferLength = MemoryLayout<BlockVertex>.stride * numVertices

    encoder.setVertexBuffer(
      buffer,
      offset: 0,
      index: BufferIndex.meshPositions.rawValue)

    var params_data = Params(
      time: getTimeSinceStart(),
      viewerPosition: gestureManager.viewerPosition,
      viewerScale: gestureManager.viewerScale,
      blocksCount: Int32(blocksCount))

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
