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

  let imagesNames: [String] = [
    "image1",
    "image2",
    "image3",
    "image4",
    "image5",
    "image6",
    "image7",
    "image8",
    "image9",
    "image10",
  ]

  // Texture array to hold all loaded images
  private var imageTextures: [MTLTexture] = []

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

    // Load the image textures
    self.loadImageTextures(device: layerRenderer.device)
  }

  /// Load images from the file system directly instead of using Asset catalog
  private func loadImageTextures(device: MTLDevice) {
    let textureLoader = MTKTextureLoader(device: device)

    print("🔍 Starting to load \(imagesNames.count) textures from Asset Catalog")

    // Try to load each image from the asset catalog
    for imageName in imagesNames {
      print("🔍 Trying to load image: \(imageName)")

      // First try to load directly from asset catalog using UIImage
      if let image = UIImage(named: imageName, in: Bundle.main, compatibleWith: nil) {
        print("✅ Found image using direct name: \(imageName)")
        loadTextureFromImage(image, name: imageName, textureLoader: textureLoader)
      }
      // Then try from myImage collection
      else if let image = UIImage(named: "myImage", in: Bundle.main, compatibleWith: nil) {
        print("✅ Found image using myImage asset: \(imageName)")
        loadTextureFromImage(image, name: imageName, textureLoader: textureLoader)
      }
      // As a fallback, look for images in the bundle as files
      else if let path = Bundle.main.path(forResource: imageName, ofType: "jpg") {
        print("✅ Found image path: \(path)")
        if let image = UIImage(contentsOfFile: path) {
          print("✅ Loaded image from path: \(path)")
          loadTextureFromImage(image, name: imageName, textureLoader: textureLoader)
        }
      } else {
        print("❌ Image \(imageName) not found in Asset Catalog")
      }
    }

    print("📊 Loaded \(imageTextures.count)/\(imagesNames.count) textures")

    // If no images were loaded, create a test checkerboard texture
    if imageTextures.isEmpty {
      print("⚠️ No textures loaded, creating a test checkerboard texture")
      createCheckerboardTexture(device: device)
    }
  }

  /// Create a simple checkerboard texture for testing when no images can be loaded
  private func createCheckerboardTexture(device: MTLDevice) {
    let width = 256
    let height = 256
    let bytesPerPixel = 4
    let bytesPerRow = bytesPerPixel * width
    let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
      pixelFormat: .rgba8Unorm,
      width: width,
      height: height,
      mipmapped: true
    )
    textureDescriptor.usage = [.shaderRead]

    guard let texture = device.makeTexture(descriptor: textureDescriptor) else {
      print("❌ Failed to create test texture")
      return
    }

    var textureData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
    for y in 0..<height {
      for x in 0..<width {
        let isCheckerboard = (x / 32 + y / 32) % 2 == 0
        let i = (y * width + x) * bytesPerPixel
        textureData[i] = isCheckerboard ? 255 : 0  // R
        textureData[i + 1] = isCheckerboard ? 0 : 255  // G
        textureData[i + 2] = 0  // B
        textureData[i + 3] = 255  // A (fully opaque)
      }
    }

    let region = MTLRegion(
      origin: MTLOrigin(x: 0, y: 0, z: 0),
      size: MTLSize(width: width, height: height, depth: 1)
    )

    texture.replace(
      region: region,
      mipmapLevel: 0,
      withBytes: textureData,
      bytesPerRow: bytesPerRow
    )

    imageTextures.append(texture)
    print("✅ Created checkerboard test texture: \(texture.width)x\(texture.height)")
  }

  private func loadTextureFromImage(_ image: UIImage, name: String, textureLoader: MTKTextureLoader)
  {
    print("🖼️ Processing image: \(name) with size: \(image.size)")
    do {
      let texture = try textureLoader.newTexture(cgImage: image.cgImage!, options: nil)
      imageTextures.append(texture)
      print(
        "✅ Successfully loaded texture for \(name) with size: \(texture.width)x\(texture.height)")
    } catch {
      print("❌ Failed to convert image to texture for \(name): \(error)")
    }
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

      // Assign proper UV coordinates for texture mapping
      cellVertices[baseIndex] = BlockVertex(
        position: p1, color: color, seed: Int32(i), height: randHeight,
        uv: SIMD2<Float>(0, 1)
      )
      cellVertices[baseIndex + 1] = BlockVertex(
        position: p2, color: color, seed: Int32(i), height: randHeight,
        uv: SIMD2<Float>(1, 1)
      )
      cellVertices[baseIndex + 2] = BlockVertex(
        position: p3, color: color, seed: Int32(i), height: randHeight,
        uv: SIMD2<Float>(1, 0)
      )
      cellVertices[baseIndex + 3] = BlockVertex(
        position: p1, color: color, seed: Int32(i), height: randHeight,
        uv: SIMD2<Float>(0, 1)
      )
      cellVertices[baseIndex + 4] = BlockVertex(
        position: p3, color: color, seed: Int32(i), height: randHeight,
        uv: SIMD2<Float>(1, 0)
      )
      cellVertices[baseIndex + 5] = BlockVertex(
        position: p4, color: color, seed: Int32(i), height: randHeight,
        uv: SIMD2<Float>(0, 0)
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

    // Debug: Check if we have any loaded textures
    print("🎨 Rendering with \(imageTextures.count) loaded textures")

    // Set texture for the fragment shader
    if !imageTextures.isEmpty {
      if imageTextures.count > 0 {
        // Use the first texture for simplicity during debugging
        let texture = imageTextures[0]
        print("🖼️ Using texture with dimensions: \(texture.width)x\(texture.height)")
        encoder.setFragmentTexture(texture, index: 0)
      } else {
        print("❌ No textures available for fragment shader")
      }
    } else {
      print("❌ No textures loaded at all")
    }

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
