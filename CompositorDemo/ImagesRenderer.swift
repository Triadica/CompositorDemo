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

private let gridSize: Int = 100

private let blocksCount: Int = gridSize * gridSize
private let verticesPerBlock = 30
private let verticesCount = verticesPerBlock * blocksCount

private let indexesPerBlock = 30  // 5 faces
private let indexesCount: Int = blocksCount * indexesPerBlock

private let blockRadius: Float = 2  // half width
private let blockHeight: Float = 20

// 定义图片文件数组
private let imageFiles = [
  "image1.jpg",
  "image2.jpg",
  "image3.jpg",
  "image4.jpg",
  "image5.jpg",
  "image6.jpg",
  "image7.jpg",
  "image8.jpg",
  "image9.jpg",
  "image10.jpg",
]

// 定义图片板的数量和顶点
private let imageRectCount = 10
private let verticesPerImageRect = 6  // 一个矩形由两个三角形组成，共6个顶点
private let imageVerticesCount = verticesPerImageRect * imageRectCount

// 图片显示的位置配置（相对于摄像机的位置）
private let imageDisplayDistance: Float = 200.0
private let imageSpacing: Float = 50.0
private let imageDefaultWidth: Float = 100.0
private let imageDefaultHeight: Float = 100.0

private struct ImageInfo {
  var width: Float
  var height: Float
  var aspectRatio: Float
  var position: SIMD3<Float>
  var rotation: Float
}

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
  var viewerRotation: Float = 0.0
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

  // 图片纹理数组
  private var imageTextures: [MTLTexture?] = Array(repeating: nil, count: imageFiles.count)
  private var imageInfos: [ImageInfo] = []
  private var imageVertexBuffer: MTLBuffer!
  private var imageRenderPipelineState: MTLRenderPipelineState & Sendable

  // 图片采样器状态
  private var imageSamplerState: MTLSamplerState!

  // 定义图片顶点结构
  private struct ImageVertex {
    var position: SIMD3<Float>
    var color: SIMD3<Float>
    var texCoord: SIMD2<Float>
    var imageIndex: Int32
  }

  init(layerRenderer: LayerRenderer) throws {
    uniformsBuffer = (0..<Renderer.maxFramesInFlight).map { _ in
      layerRenderer.device.makeBuffer(length: MemoryLayout<PathProperties>.uniformStride)!
    }

    renderPipelineState = try Self.makeRenderPipelineDescriptor(layerRenderer: layerRenderer)
    imageRenderPipelineState = try ImagesRenderer.makeImageRenderPipelineState(
      layerRenderer: layerRenderer)

    self.computeDevice = MTLCreateSystemDefaultDevice()!
    let library = computeDevice.makeDefaultLibrary()!
    let cellUpdateBase: any MTLFunction = library.makeFunction(name: "imagesComputeShader")!
    computePipeLine = try computeDevice.makeComputePipelineState(function: cellUpdateBase)

    computeCommandQueue = computeDevice.makeCommandQueue()!

    // 加载图片纹理
    loadImageTextures(device: layerRenderer.device)
    // 创建图片顶点缓冲区
    createImageVertexBuffer(device: layerRenderer.device)

    self.createImagesVerticesBuffer(device: layerRenderer.device)
    self.createImagesIndexBuffer(device: layerRenderer.device)
    self.createImagesComputeBuffer(device: layerRenderer.device)
  }

  private static func makeImageRenderPipelineState(layerRenderer: LayerRenderer) throws
    -> MTLRenderPipelineState
  {
    let pipelineDescriptor = Renderer.defaultRenderPipelineDescriptor(
      layerRenderer: layerRenderer)

    let library = layerRenderer.device.makeDefaultLibrary()!

    // Check if functions exist before trying to use them
    let vertexFunction = library.makeFunction(name: "imageVertexShader")
    let fragmentFunction = library.makeFunction(name: "imageFragmentShader")

    pipelineDescriptor.vertexFunction = vertexFunction
    pipelineDescriptor.fragmentFunction = fragmentFunction

    pipelineDescriptor.label = "ImageRenderPipeline"
    pipelineDescriptor.vertexDescriptor = ImagesRenderer.buildMetalImageVertexDescriptor()

    pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
    return try layerRenderer.device.makeRenderPipelineState(descriptor: pipelineDescriptor)
  }

  class func buildMetalImageVertexDescriptor() -> MTLVertexDescriptor {
    let vertexDescriptor = MTLVertexDescriptor()

    vertexDescriptor.attributes[0].format = .float3
    vertexDescriptor.attributes[0].offset = 0
    vertexDescriptor.attributes[0].bufferIndex = 0

    vertexDescriptor.attributes[1].format = .float3
    vertexDescriptor.attributes[1].offset = MemoryLayout<SIMD3<Float>>.stride
    vertexDescriptor.attributes[1].bufferIndex = 0

    vertexDescriptor.attributes[2].format = .float2
    vertexDescriptor.attributes[2].offset = MemoryLayout<SIMD3<Float>>.stride * 2
    vertexDescriptor.attributes[2].bufferIndex = 0

    vertexDescriptor.attributes[3].format = .int
    vertexDescriptor.attributes[3].offset =
      MemoryLayout<SIMD3<Float>>.stride * 2
      + MemoryLayout<SIMD2<Float>>.stride
    vertexDescriptor.attributes[3].bufferIndex = 0

    vertexDescriptor.layouts[0].stride = MemoryLayout<ImageVertex>.stride

    return vertexDescriptor
  }

  /// create and sets the vertices of the lamp
  private func createImagesVerticesBuffer(device: MTLDevice) {
    let bufferLength = MemoryLayout<BlockVertex>.stride * verticesCount
    vertexBuffer = device.makeBuffer(length: bufferLength)!
    vertexBuffer.label = "Lamp vertex buffer"
    var cellVertices: UnsafeMutablePointer<BlockVertex> {
      vertexBuffer.contents().assumingMemoryBound(to: BlockVertex.self)
    }

    let unit = 5

    for i in 0..<gridSize {
      for j in 0..<gridSize {
        let idx = i * gridSize + j
        // Random color for each lamp
        let red = Float.random(in: 0.2...0.99)
        let g = Float.random(in: 0.2...0.99)
        let b = Float.random(in: 0.2...0.99)
        var color = SIMD3<Float>(red, g, b)

        let baseIndex = idx * verticesPerBlock
        var randHeight = pow(Float.random(in: 0.1...1), 3) * blockHeight
        var r: Float = blockRadius * Float.random(in: 0.5...1.0)
        if i % unit == 0 || j % unit == 0 {
          randHeight = 0.02
          r = blockRadius
          color = SIMD3<Float>(0.1, 0.1, 0.1)
        }

        let p1: SIMD3<Float> = SIMD3<Float>(-r, 0, -r)
        let p2: SIMD3<Float> = SIMD3<Float>(r, 0, -r)
        let p3: SIMD3<Float> = SIMD3<Float>(r, 0, r)
        let p4: SIMD3<Float> = SIMD3<Float>(-r, 0, r)
        let p5: SIMD3<Float> = SIMD3<Float>(-r, randHeight, -r)
        let p6: SIMD3<Float> = SIMD3<Float>(r, randHeight, -r)
        let p7: SIMD3<Float> = SIMD3<Float>(r, randHeight, r)
        let p8: SIMD3<Float> = SIMD3<Float>(-r, randHeight, r)

        // front face, 126,165
        cellVertices[baseIndex] = BlockVertex(
          position: p1, color: color, seed: Int32(idx), height: randHeight,
          uv: SIMD2<Float>(0, 0)
        )
        cellVertices[baseIndex + 1] = BlockVertex(
          position: p2, color: color, seed: Int32(idx), height: randHeight,
          uv: SIMD2<Float>(2 * r, 0)
        )
        cellVertices[baseIndex + 2] = BlockVertex(
          position: p6, color: color, seed: Int32(idx), height: randHeight,
          uv: SIMD2<Float>(2 * r, randHeight)
        )
        cellVertices[baseIndex + 3] = BlockVertex(
          position: p1, color: color, seed: Int32(idx), height: randHeight,
          uv: SIMD2<Float>(0, 0)
        )
        cellVertices[baseIndex + 4] = BlockVertex(
          position: p6, color: color, seed: Int32(idx), height: randHeight,
          uv: SIMD2<Float>(2 * r, randHeight)
        )
        cellVertices[baseIndex + 5] = BlockVertex(
          position: p5, color: color, seed: Int32(idx), height: randHeight,
          uv: SIMD2<Float>(0, randHeight)
        )
        // right face, 237,276
        cellVertices[baseIndex + 6] = BlockVertex(
          position: p2, color: color, seed: Int32(idx), height: randHeight, uv: SIMD2<Float>(0, 0))
        cellVertices[baseIndex + 7] = BlockVertex(
          position: p3, color: color, seed: Int32(idx), height: randHeight,
          uv: SIMD2<Float>(2 * r, 0))
        cellVertices[baseIndex + 8] = BlockVertex(
          position: p7, color: color, seed: Int32(idx), height: randHeight,
          uv: SIMD2<Float>(2 * r, randHeight))
        cellVertices[baseIndex + 9] = BlockVertex(
          position: p2, color: color, seed: Int32(idx), height: randHeight, uv: SIMD2<Float>(0, 0))
        cellVertices[baseIndex + 10] = BlockVertex(
          position: p7, color: color, seed: Int32(idx), height: randHeight,
          uv: SIMD2<Float>(2 * r, randHeight))
        cellVertices[baseIndex + 11] = BlockVertex(
          position: p6, color: color, seed: Int32(idx), height: randHeight,
          uv: SIMD2<Float>(0, randHeight))
        // back face, 348,387
        cellVertices[baseIndex + 12] = BlockVertex(
          position: p3, color: color, seed: Int32(idx), height: randHeight, uv: SIMD2<Float>(0, 0))
        cellVertices[baseIndex + 13] = BlockVertex(
          position: p4, color: color, seed: Int32(idx), height: randHeight,
          uv: SIMD2<Float>(2 * r, 0))
        cellVertices[baseIndex + 14] = BlockVertex(
          position: p8, color: color, seed: Int32(idx), height: randHeight,
          uv: SIMD2<Float>(2 * r, randHeight))
        cellVertices[baseIndex + 15] = BlockVertex(
          position: p3, color: color, seed: Int32(idx), height: randHeight, uv: SIMD2<Float>(0, 0))
        cellVertices[baseIndex + 16] = BlockVertex(
          position: p8, color: color, seed: Int32(idx), height: randHeight,
          uv: SIMD2<Float>(2 * r, randHeight))
        cellVertices[baseIndex + 17] = BlockVertex(
          position: p7, color: color, seed: Int32(idx), height: randHeight,
          uv: SIMD2<Float>(0, randHeight))
        // left face, 415,458
        cellVertices[baseIndex + 18] = BlockVertex(
          position: p4, color: color, seed: Int32(idx), height: randHeight, uv: SIMD2<Float>(0, 0))
        cellVertices[baseIndex + 19] = BlockVertex(
          position: p1, color: color, seed: Int32(idx), height: randHeight,
          uv: SIMD2<Float>(2 * r, 0))
        cellVertices[baseIndex + 20] = BlockVertex(
          position: p5, color: color, seed: Int32(idx), height: randHeight,
          uv: SIMD2<Float>(2 * r, randHeight))
        cellVertices[baseIndex + 21] = BlockVertex(
          position: p4, color: color, seed: Int32(idx), height: randHeight, uv: SIMD2<Float>(0, 0))
        cellVertices[baseIndex + 22] = BlockVertex(
          position: p5, color: color, seed: Int32(idx), height: randHeight,
          uv: SIMD2<Float>(2 * r, randHeight))
        cellVertices[baseIndex + 23] = BlockVertex(
          position: p8, color: color, seed: Int32(idx), height: randHeight,
          uv: SIMD2<Float>(0, randHeight))
        // top face, 567,578
        cellVertices[baseIndex + 24] = BlockVertex(
          position: p5, color: color, seed: Int32(idx), height: randHeight, uv: SIMD2<Float>(0, 0))
        cellVertices[baseIndex + 25] = BlockVertex(
          position: p6, color: color, seed: Int32(idx), height: randHeight, uv: SIMD2<Float>(0, 0))
        cellVertices[baseIndex + 26] = BlockVertex(
          position: p7, color: color, seed: Int32(idx), height: randHeight, uv: SIMD2<Float>(0, 0))
        cellVertices[baseIndex + 27] = BlockVertex(
          position: p5, color: color, seed: Int32(idx), height: randHeight, uv: SIMD2<Float>(0, 0))
        cellVertices[baseIndex + 28] = BlockVertex(
          position: p7, color: color, seed: Int32(idx), height: randHeight, uv: SIMD2<Float>(0, 0))
        cellVertices[baseIndex + 29] = BlockVertex(
          position: p8, color: color, seed: Int32(idx), height: randHeight, uv: SIMD2<Float>(0, 0))

      }
    }

  }

  func resetComputeState() {
    self.createImagesComputeBuffer(device: computeDevice)
  }

  private func createImagesIndexBuffer(device: MTLDevice) {
    let bufferLength = MemoryLayout<UInt32>.stride * indexesCount
    indexBuffer = device.makeBuffer(length: bufferLength)!
    indexBuffer.label = "Lamp index buffer"

    let cellIndices = indexBuffer.contents().bindMemory(
      to: UInt32.self, capacity: indexesCount)
    let total = blocksCount * 30
    for i in 0..<total {
      cellIndices[i] = UInt32(i)
    }

  }

  private func createImagesComputeBuffer(device: MTLDevice) {
    let bufferLength = MemoryLayout<CellBase>.stride * blocksCount

    computeBuffer = PingPongBuffer(device: device, length: bufferLength)

    guard let computeBuffer = computeBuffer else {
      print("Failed to create compute buffer")
      return
    }
    computeBuffer.addLabel("Lamp compute buffer")

    let contents = computeBuffer.currentBuffer.contents()
    let blocksBase = contents.bindMemory(to: CellBase.self, capacity: blocksCount)

    let middle = Float(gridSize) / 2.0

    for i in 0..<gridSize {
      for j in 0..<gridSize {
        let xOffset = (Float(i) - middle) * blockRadius * 2
        let zOffset = (Float(j) - middle) * blockRadius * 2
        let yOffset: Float = 0.0

        let lampPosition = SIMD3<Float>(xOffset, yOffset, zOffset)
        // Random color for each lamp
        let r = Float.random(in: 0.1...1.0)
        let g = Float.random(in: 0.1...1.0)
        let b = Float.random(in: 0.1...1.0)
        let color = SIMD3<Float>(r, g, b)

        blocksBase[i * gridSize + j] = CellBase(
          position: lampPosition, color: color, blockIdf: Float(i), velocity: .zero)
      }
    }

    computeBuffer.copy_to_next()
  }

  private func loadImageTextures(device: MTLDevice) {
    let textureLoader = MTKTextureLoader(device: device)

    // 创建采样器状态
    let samplerDescriptor = MTLSamplerDescriptor()
    samplerDescriptor.minFilter = .linear
    samplerDescriptor.magFilter = .linear
    samplerDescriptor.mipFilter = .linear
    samplerDescriptor.normalizedCoordinates = true
    samplerDescriptor.supportArgumentBuffers = true
    imageSamplerState = device.makeSamplerState(descriptor: samplerDescriptor)!

    // 重置图片信息数组
    imageInfos.removeAll()

    // 围绕摄像机在四周布置图片
    let positions: [SIMD3<Float>] = [
      SIMD3<Float>(0, 0, imageDisplayDistance),  // 前方
      SIMD3<Float>(imageDisplayDistance, 0, imageDisplayDistance),  // 右前方
      SIMD3<Float>(imageDisplayDistance, 0, 0),  // 右方
      SIMD3<Float>(imageDisplayDistance, 0, -imageDisplayDistance),  // 右后方
      SIMD3<Float>(0, 0, -imageDisplayDistance),  // 后方
      SIMD3<Float>(-imageDisplayDistance, 0, -imageDisplayDistance),  // 左后方
      SIMD3<Float>(-imageDisplayDistance, 0, 0),  // 左方
      SIMD3<Float>(-imageDisplayDistance, 0, imageDisplayDistance),  // 左前方
      SIMD3<Float>(0, imageDisplayDistance, 0),  // 上方
      SIMD3<Float>(0, -imageDisplayDistance, 0),  // 下方
    ]

    let rotations: [Float] = [
      0.0,  // 前方
      Float.pi / 4,  // 右前方
      Float.pi / 2,  // 右方
      3 * Float.pi / 4,  // 右后方
      Float.pi,  // 后方
      5 * Float.pi / 4,  // 左后方
      3 * Float.pi / 2,  // 左方
      7 * Float.pi / 4,  // 左前方
      -Float.pi / 2,  // 上方
      Float.pi / 2,  // 下方
    ]

    // 加载每个图片并创建纹理
    for i in 0..<min(imageFiles.count, imageRectCount) {
      if let image = UIImage(named: imageFiles[i]) {
        do {
          // 加载图片纹理
          let options: [MTKTextureLoader.Option: Any] = [
            .SRGB: false,
            .generateMipmaps: true,
          ]
          imageTextures[i] = try textureLoader.newTexture(cgImage: image.cgImage!, options: options)

          // 根据图片纹理的尺寸计算宽高比
          let width = Float(imageTextures[i]!.width)
          let height = Float(imageTextures[i]!.height)
          let aspectRatio = width / height

          // 保持宽高比，调整显示尺寸
          var displayWidth = imageDefaultWidth
          var displayHeight = imageDefaultHeight

          if aspectRatio > 1.0 {
            // 宽图
            displayHeight = displayWidth / aspectRatio
          } else {
            // 高图
            displayWidth = displayHeight * aspectRatio
          }

          // 创建图片信息
          let imageInfo = ImageInfo(
            width: displayWidth,
            height: displayHeight,
            aspectRatio: aspectRatio,
            position: positions[i],
            rotation: rotations[i]
          )
          imageInfos.append(imageInfo)

        } catch {
          print("Failed to load texture for image \(imageFiles[i]): \(error)")
          // 使用默认尺寸
          let imageInfo = ImageInfo(
            width: imageDefaultWidth,
            height: imageDefaultHeight,
            aspectRatio: 1.0,
            position: positions[i],
            rotation: rotations[i]
          )
          imageInfos.append(imageInfo)
        }
      } else {
        print("Image not found: \(imageFiles[i])")
        // 使用默认尺寸
        let imageInfo = ImageInfo(
          width: imageDefaultWidth,
          height: imageDefaultHeight,
          aspectRatio: 1.0,
          position: positions[i],
          rotation: rotations[i]
        )
        imageInfos.append(imageInfo)
      }
    }
  }

  private func createImageVertexBuffer(device: MTLDevice) {
    // 创建图片矩形顶点缓冲区
    let bufferLength = MemoryLayout<ImageVertex>.stride * imageVerticesCount
    imageVertexBuffer = device.makeBuffer(length: bufferLength)!
    imageVertexBuffer.label = "Image rectangle vertex buffer"

    // 确保已加载图片纹理和信息
    if imageInfos.isEmpty {
      loadImageTextures(device: device)
    }

    // 获取顶点缓冲区内存指针
    let imageVertices = imageVertexBuffer.contents().assumingMemoryBound(to: ImageVertex.self)

    // 为每个图片创建一个矩形（由两个三角形组成）
    for i in 0..<imageInfos.count {
      let info = imageInfos[i]
      let baseIndex = i * verticesPerImageRect

      // 计算矩形的四个顶点
      // 考虑到旋转，我们需要绕y轴旋转
      let halfWidth = info.width / 2.0
      let halfHeight = info.height / 2.0

      let cosR = cos(info.rotation)
      let sinR = sin(info.rotation)

      // 变换坐标以应用旋转
      // 左下角
      let p1 = SIMD3<Float>(
        -halfWidth * cosR - 0 * sinR + info.position.x,
        -halfHeight + info.position.y,
        -halfWidth * sinR + 0 * cosR + info.position.z
      )

      // 右下角
      let p2 = SIMD3<Float>(
        halfWidth * cosR - 0 * sinR + info.position.x,
        -halfHeight + info.position.y,
        halfWidth * sinR + 0 * cosR + info.position.z
      )

      // 右上角
      let p3 = SIMD3<Float>(
        halfWidth * cosR - (2 * halfHeight) * sinR + info.position.x,
        halfHeight + info.position.y,
        halfWidth * sinR + (2 * halfHeight) * cosR + info.position.z
      )

      // 左上角
      let p4 = SIMD3<Float>(
        -halfWidth * cosR - (2 * halfHeight) * sinR + info.position.x,
        halfHeight + info.position.y,
        -halfWidth * sinR + (2 * halfHeight) * cosR + info.position.z
      )

      // 通用颜色（全白，让纹理颜色显示）
      let color = SIMD3<Float>(1.0, 1.0, 1.0)

      // 第一个三角形 (p1, p2, p3)
      imageVertices[baseIndex] = ImageVertex(
        position: p1,
        color: color,
        texCoord: SIMD2<Float>(0.0, 1.0),
        imageIndex: Int32(i)
      )

      imageVertices[baseIndex + 1] = ImageVertex(
        position: p2,
        color: color,
        texCoord: SIMD2<Float>(1.0, 1.0),
        imageIndex: Int32(i)
      )

      imageVertices[baseIndex + 2] = ImageVertex(
        position: p3,
        color: color,
        texCoord: SIMD2<Float>(1.0, 0.0),
        imageIndex: Int32(i)
      )

      // 第二个三角形 (p1, p3, p4)
      imageVertices[baseIndex + 3] = ImageVertex(
        position: p1,
        color: color,
        texCoord: SIMD2<Float>(0.0, 1.0),
        imageIndex: Int32(i)
      )

      imageVertices[baseIndex + 4] = ImageVertex(
        position: p3,
        color: color,
        texCoord: SIMD2<Float>(1.0, 0.0),
        imageIndex: Int32(i)
      )

      imageVertices[baseIndex + 5] = ImageVertex(
        position: p4,
        color: color,
        texCoord: SIMD2<Float>(0.0, 0.0),
        imageIndex: Int32(i)
      )
    }
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
    pipelineDescriptor.vertexDescriptor = BlocksRenderer.buildMetalVertexDescriptor()

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
      viewerRotation: gestureManager.viewerRotation)
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
    // 渲染方块
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
      viewerRotation: gestureManager.viewerRotation)

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

    // 渲染图片
    renderImages(
      encoder: encoder, drawable: drawable, device: device, tintValue: tintValue, params: params)
  }

  private func renderImages(
    encoder: MTLRenderCommandEncoder,
    drawable: LayerRenderer.Drawable,
    device: MTLDevice,
    tintValue: Float,
    params: MTLBuffer
  ) {
    // 设置渲染管道状态
    encoder.setRenderPipelineState(imageRenderPipelineState)

    // 设置Uniform数据
    var demoUniform = TintUniforms(tintOpacity: tintValue)
    encoder.setVertexBytes(
      &demoUniform,
      length: MemoryLayout<TintUniforms>.size,
      index: BufferIndex.tintUniforms.rawValue)

    // 设置顶点缓冲区
    encoder.setVertexBuffer(
      imageVertexBuffer,
      offset: 0,
      index: BufferIndex.meshPositions.rawValue)

    // 设置参数缓冲区
    encoder.setVertexBuffer(
      params,
      offset: 0,
      index: BufferIndex.params.rawValue)

    // 设置纹理采样器
    encoder.setFragmentSamplerState(imageSamplerState, index: 0)

    // 设置纹理
    for i in 0..<min(imageTextures.count, imageRectCount) {
      if let texture = imageTextures[i] {
        encoder.setFragmentTexture(texture, index: i)
      }
    }

    // 绘制图片矩形
    let imageVerticesCount = min(imageInfos.count, imageRectCount) * verticesPerImageRect
    if imageVerticesCount > 0 {
      encoder.drawPrimitives(
        type: .triangle,
        vertexStart: 0,
        vertexCount: imageVerticesCount
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
    for event in events {
      gestureManager.onSpatialEvent(event: event)
    }
  }
}
