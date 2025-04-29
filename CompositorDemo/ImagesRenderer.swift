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

private let imagesCount: Int = 8

private let verticesPerBlock = 6
private let verticesCount = verticesPerBlock * imagesCount

private let indexesPerBlock = 6
private let indexesCount: Int = imagesCount * indexesPerBlock

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
  var imagesCount: Int32 = 0
  var _padding: SIMD3<Float> = .zero  // required for 48 bytes alignment
}

@MainActor
class ImagesRenderer: CustomRenderer {
  private let renderPipelineState: MTLRenderPipelineState & Sendable

  private var uniformsBuffer: [MTLBuffer]
  /// a buffer to hold the vertices of the lamp
  var vertexBuffer: MTLBuffer!

  var indexBuffer: MTLBuffer!

  // Define a struct to store image information including dimensions
  private struct ImageInfo {
    var texture: MTLTexture
    var width: Float
    var height: Float
    var aspectRatio: Float

    init(texture: MTLTexture) {
      self.texture = texture
      self.width = Float(texture.width)
      self.height = Float(texture.height)
      self.aspectRatio = self.width / self.height
    }
  }

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

  // Change to store ImageInfo instead of just textures
  private var imageInfos: [ImageInfo] = []

  let computeDevice: MTLDevice
  var computeBuffer: PingPongBuffer?
  let computePipeLine: MTLComputePipelineState
  let computeCommandQueue: MTLCommandQueue

  var gestureManager: GestureManager = GestureManager(onScene: true)
  var selectedImage: (Int, SIMD3<Float>)?

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

    // Load the image textures
    Task {
      await self.loadImageTextures(device: layerRenderer.device)
      // Recreate vertices after images are loaded
      DispatchQueue.main.async {
        self.createBlocksVerticesBuffer(device: layerRenderer.device)
      }
    }

    self.createBlocksVerticesBuffer(device: layerRenderer.device)
    self.createBlocksIndexBuffer(device: layerRenderer.device)
    self.createBlocksComputeBuffer(device: layerRenderer.device)
  }

  /// Load images from the file system directly instead of using Asset catalog
  private func loadImageTextures(device: MTLDevice) async {
    let textureLoader = MTKTextureLoader(device: device)

    print("🔍 Starting to load \(imagesNames.count) textures from Asset Catalog")

    // Try to load each image from the asset catalog
    for imageName in imagesNames {
      print("🔍 Trying to load image: \(imageName)")
      let url = "https://repo.webgpu.art/image-assets/\(imageName).jpg"
      do {
        // Asynchronously load the image from URL
        let imageURL = URL(string: url)!
        let (data, _) = try await URLSession.shared.data(from: imageURL)

        if let image = UIImage(data: data) {
          print("✅ Successfully downloaded image from: \(url)")
          loadTextureFromImage(image, name: imageName, textureLoader: textureLoader)
        } else {
          print("❌ Failed to create UIImage from data for: \(url)")
        }
      } catch {
        print("❌ Failed to load image from URL: \(url) - Error: \(error.localizedDescription)")

        // Fallback to local assets if URL fails
        if let image = UIImage(named: imageName, in: Bundle.main, compatibleWith: nil) {
          print("✅ Fallback: Found image in asset catalog: \(imageName)")
          loadTextureFromImage(image, name: imageName, textureLoader: textureLoader)
        } else if let path = Bundle.main.path(forResource: imageName, ofType: "jpg") {
          if let image = UIImage(contentsOfFile: path) {
            print("✅ Fallback: Loaded image from file path: \(path)")
            loadTextureFromImage(image, name: imageName, textureLoader: textureLoader)
          }
        }
      }
    }

    print("📊 Loaded \(imageInfos.count)/\(imagesNames.count) textures")

    // If no images were loaded, create a test checkerboard texture
    if imageInfos.isEmpty {
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

    imageInfos.append(ImageInfo(texture: texture))
    print("✅ Created checkerboard test texture: \(texture.width)x\(texture.height)")
  }

  private func loadTextureFromImage(_ image: UIImage, name: String, textureLoader: MTKTextureLoader)
  {
    print("🖼️ Processing image: \(name) with size: \(image.size)")
    do {
      let texture = try textureLoader.newTexture(cgImage: image.cgImage!, options: nil)
      imageInfos.append(ImageInfo(texture: texture))
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
    vertexBuffer.label = "Images vertex buffer"
    var cellVertices: UnsafeMutablePointer<BlockVertex> {
      vertexBuffer.contents().assumingMemoryBound(to: BlockVertex.self)
    }

    for i in 0..<imagesCount {
      let color = SIMD3<Float>(1, 1, 1)  // White to preserve image colors

      let baseIndex = i * 6

      // Default dimensions if no image is available
      var width: Float = 0.4
      var height: Float = 0.4

      // Use actual image dimensions if available
      if i < imageInfos.count {
        // Scale dimensions to a reasonable size while maintaining aspect ratio
        let imageInfo = imageInfos[i]

        width = imageInfo.width / 1000.0
        height = imageInfo.height / 1000.0
      }

      // Create vertices for a rectangle with proper dimensions
      let halfWidth = width / 2
      let halfHeight = height / 2

      let p1: SIMD3<Float> = SIMD3<Float>(-halfWidth, -halfHeight, 0)
      let p2: SIMD3<Float> = SIMD3<Float>(halfWidth, -halfHeight, 0)
      let p3: SIMD3<Float> = SIMD3<Float>(halfWidth, halfHeight, 0)
      let p4: SIMD3<Float> = SIMD3<Float>(-halfWidth, halfHeight, 0)

      // Assign proper UV coordinates for texture mapping (flipped vertically for Metal's coordinate system)
      cellVertices[baseIndex] = BlockVertex(
        position: p1, color: color, seed: Int32(i), height: height,
        uv: SIMD2<Float>(0, 1)
      )
      cellVertices[baseIndex + 1] = BlockVertex(
        position: p2, color: color, seed: Int32(i), height: height,
        uv: SIMD2<Float>(1, 1)
      )
      cellVertices[baseIndex + 2] = BlockVertex(
        position: p3, color: color, seed: Int32(i), height: height,
        uv: SIMD2<Float>(1, 0)
      )
      cellVertices[baseIndex + 3] = BlockVertex(
        position: p1, color: color, seed: Int32(i), height: height,
        uv: SIMD2<Float>(0, 1)
      )
      cellVertices[baseIndex + 4] = BlockVertex(
        position: p3, color: color, seed: Int32(i), height: height,
        uv: SIMD2<Float>(1, 0)
      )
      cellVertices[baseIndex + 5] = BlockVertex(
        position: p4, color: color, seed: Int32(i), height: height,
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
    let total = imagesCount * 6
    for i in 0..<total {
      cellIndices[i] = UInt32(i)
    }

  }

  private func createBlocksComputeBuffer(device: MTLDevice) {
    let bufferLength = MemoryLayout<CellBase>.stride * imagesCount

    computeBuffer = PingPongBuffer(device: device, length: bufferLength)

    guard let computeBuffer = computeBuffer else {
      print("Failed to create compute buffer")
      return
    }
    computeBuffer.addLabel("Lamp compute buffer")

    let contents = computeBuffer.currentBuffer.contents()
    let blocksBase = contents.bindMemory(to: CellBase.self, capacity: imagesCount)

    for i in 0..<imagesCount {

      let angle = Float(i) * (2.0 * .pi / Float(imagesCount))
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
      imagesCount: Int32(imagesCount))
    computeEncoder.setBytes(&params, length: MemoryLayout<Params>.size, index: 2)
    let threadGroupSize = min(computePipeLine.maxTotalThreadsPerThreadgroup, 256)
    let threadsPerThreadgroup = MTLSize(width: threadGroupSize, height: 1, depth: 1)
    let threadGroups = MTLSize(
      width: (imagesCount + threadGroupSize - 1) / threadGroupSize,
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
      imagesCount: Int32(imagesCount))

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

    // Set texture for the fragment shader
    if !imageInfos.isEmpty {
      // Divide the vertices into blocks and draw each block with its corresponding texture
      for i in 0..<min(imagesCount, imageInfos.count) {
        let texture = imageInfos[i].texture
        encoder.setFragmentTexture(texture, index: 0)

        // Calculate the index range for this block
        let startIndex = i * indexesPerBlock
        let indexCount = indexesPerBlock

        encoder.drawIndexedPrimitives(
          type: .triangle,
          indexCount: indexCount,
          indexType: .uint32,
          indexBuffer: indexBuffer,
          indexBufferOffset: startIndex * MemoryLayout<UInt32>.size
        )
      }

      // If we have more blocks than textures, use the last texture for remaining blocks
      if imagesCount > imageInfos.count {
        let lastTexture = imageInfos.last!.texture
        encoder.setFragmentTexture(lastTexture, index: 0)

        // Calculate the index range for remaining blocks
        let startIndex = imageInfos.count * indexesPerBlock
        let indexCount = (imagesCount - imageInfos.count) * indexesPerBlock

        if indexCount > 0 {
          encoder.drawIndexedPrimitives(
            type: .triangle,
            indexCount: indexCount,
            indexType: .uint32,
            indexBuffer: indexBuffer,
            indexBufferOffset: startIndex * MemoryLayout<UInt32>.size
          )
        }
      }
    } else {
      print("❌ No textures loaded at all")
      // Draw without textures as fallback
      encoder.drawIndexedPrimitives(
        type: .triangle,
        indexCount: indexesCount,
        indexType: .uint32,
        indexBuffer: indexBuffer,
        indexBufferOffset: 0
      )
    }
  }

  func updateUniformBuffers(
    _ drawCommand: TintDrawCommand,
    drawable: LayerRenderer.Drawable
  ) {
    drawCommand.uniforms.contents().assumingMemoryBound(to: Uniforms.self).pointee = Uniforms(
      drawable: drawable)
  }

  func onSpatialEvents(events: SpatialEventCollection) {
    // read array of Cell s from computeBuffer

    let cells = computeBuffer!.currentBuffer.contents().bindMemory(
      to: CellBase.self, capacity: imagesCount)

    for event in events {
      if event.phase == .ended {
        self.selectedImage = nil
      } else {
        if let (idx, prevPosition) = self.selectedImage {
          let position = event.inputDevicePose!.pose3D.position.to_simd3
          let delta = position - prevPosition
          let cell = cells[idx]
          let newPosition = cell.position + delta * 5
          cells[idx].position = newPosition
          self.selectedImage = (idx, position)
        } else {
          self.selectedImage = nil

          if let ray = event.selectionRay {

            let origin = ray.origin.to_simd3
            let direction = ray.direction.to_simd3

            // Calculate which image is being hit by the ray
            var closestHitDistance: Float = Float.greatestFiniteMagnitude
            var hitImageIndex: Int = -1

            // Iterate through all images to find the closest hit
            for i in 0..<imagesCount {
              let cell = cells[i]
              let imagePosition = cell.position

              // Assuming images are facing the camera (plane perpendicular to camera direction)
              // Calculate the plane the image lies on
              let planeNormal = normalize(
                imagePosition - origin
              )  // Simplified - assumes image faces viewer

              // Get image dimensions if available
              let halfWidth: Float = i < imageInfos.count ? (imageInfos[i].width / 2000.0) : 0.2
              let halfHeight: Float = i < imageInfos.count ? (imageInfos[i].height / 2000.0) : 0.2

              // Ray-plane intersection formula: t = dot(planePos - rayOrigin, planeNormal) / dot(rayDirection, planeNormal)
              let denominator = dot(direction, planeNormal)

              // Skip if ray is parallel to the plane
              if abs(denominator) > 0.0001 {
                let t = dot(imagePosition - origin, planeNormal) / denominator

                // Only consider hits in front of the viewer
                if t > 0 {
                  // Calculate intersection point
                  let hitPoint = origin + t * direction

                  // Calculate local coordinates on the image plane
                  let localUp = SIMD3<Float>(0, 1, 0)
                  let localRight = normalize(cross(localUp, planeNormal))
                  let localUp2 = cross(planeNormal, localRight)

                  let localX = dot(hitPoint - imagePosition, localRight)
                  let localY = dot(hitPoint - imagePosition, localUp2)

                  // Check if hit point is within the image bounds
                  if abs(localX) <= halfWidth && abs(localY) <= halfHeight {
                    if t < closestHitDistance {
                      closestHitDistance = t
                      hitImageIndex = i
                    }
                  }
                }
              }
            }

            if hitImageIndex >= 0 {
              self.selectedImage = (hitImageIndex, event.inputDevicePose!.pose3D.position.to_simd3)
            }
          }
        }
      }
    }

  }
}

extension Point3D {
  /// turn into SIMD3
  fileprivate var to_simd3: SIMD3<Float> {
    return SIMD3<Float>(Float(x), Float(y), Float(z))
  }
}

extension Vector3D {
  fileprivate var to_simd3: SIMD3<Float> {
    return SIMD3<Float>(Float(x), Float(y), Float(z))
  }
}
