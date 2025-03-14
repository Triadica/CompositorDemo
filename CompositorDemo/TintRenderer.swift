/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
A renderer that displays a set of color swatches.
*/

import CompositorServices
import Metal
import MetalKit
import Spatial
import simd

let maxFramesInFlight = 3

let lampCount: Int = 10
let verticesPerLamp: Int = 48  // 8 rectangles * 6 vertices per rectangle
let numVertices: Int = lampCount * verticesPerLamp

@MainActor
class TintRenderer {
    private let renderPipelineState: MTLRenderPipelineState & Sendable

    private var uniformsBuffer: [MTLBuffer]
    /// a buffer to hold the vertices of the lamp
    var lampVerticesBuffer: MTLBuffer!

    init(layerRenderer: LayerRenderer) throws {
        uniformsBuffer = (0..<Renderer.maxFramesInFlight).map { _ in
            layerRenderer.device.makeBuffer(length: MemoryLayout<PathProperties>.uniformStride)!
        }

        renderPipelineState = try Self.makePipelineDescriptor(layerRenderer: layerRenderer)
        self.createLampVerticesBuffer(device: layerRenderer.device)
    }

    /// create and sets the vertices of the lamp
    private func createLampVerticesBuffer(device: MTLDevice) {
        let bufferLength = MemoryLayout<Vertex>.stride * numVertices
        lampVerticesBuffer = device.makeBuffer(length: bufferLength)!
        lampVerticesBuffer.label = "Lamp vertex buffer"
        var lampVertices: UnsafeMutablePointer<Vertex> {
            lampVerticesBuffer.contents().assumingMemoryBound(to: Vertex.self)
        }

        let horizontalScale: Float = 6.0
        let verticalScale: Float = 2.0
        let depth: Float = -6.0

        for i in 0..<lampCount {
            // Random position offsets for each lamp
            let xOffset = Float.random(in: -5...5)
            let yOffset = Float.random(in: -5...5)
            let zOffset = Float.random(in: -2...2)

            let lampPosition = SIMD3<Float>(xOffset, yOffset, zOffset)
            // Random color for each lamp
            let r = Float.random(in: 0.5...1.0)
            let g = Float.random(in: 0.0...0.5)
            let b = Float.random(in: 0.1...0.8)
            let roseColor = SIMD3<Float>(r, g, b)
            let baseIndex = i * verticesPerLamp
            let petals = 8
            let radius: Float = 0.5

            for p in 0..<petals {
                let angle = Float(p) * (2 * Float.pi / Float(petals))
                let nextAngle = Float(p + 1) * (2 * Float.pi / Float(petals))

                // Calculate the four corners of this rectangular petal
                let innerPoint1 = SIMD3<Float>(
                    cos(angle) * radius, sin(angle) * radius, depth)
                let innerPoint2 = SIMD3<Float>(
                    cos(nextAngle) * radius, sin(nextAngle) * radius, depth)
                let outerPoint1 = SIMD3<Float>(
                    cos(angle) * radius * 2, sin(angle) * radius * 2, depth)
                let outerPoint2 = SIMD3<Float>(
                    cos(nextAngle) * radius * 2, sin(nextAngle) * radius * 2, depth)

                let vertexBase = baseIndex + p * 6

                // First triangle of rectangle (inner1, outer1, inner2)
                lampVertices[vertexBase] = Vertex(
                    position: innerPoint1 + lampPosition, color: roseColor)
                lampVertices[vertexBase + 1] = Vertex(
                    position: outerPoint1 + lampPosition,
                    color: roseColor
                )
                lampVertices[vertexBase + 2] = Vertex(
                    position: innerPoint2 + lampPosition,
                    color: roseColor
                )

                // Second triangle of rectangle (inner2, outer1, outer2)
                lampVertices[vertexBase + 3] = Vertex(
                    position: innerPoint2 + lampPosition,
                    color: roseColor
                )
                lampVertices[vertexBase + 4] = Vertex(
                    position: outerPoint1 + lampPosition,
                    color: roseColor
                )
                lampVertices[vertexBase + 5] = Vertex(
                    position: outerPoint2 + lampPosition,
                    color: roseColor
                )
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
            MemoryLayout<Vertex>.stride
        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepRate = 1
        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepFunction =
            MTLVertexStepFunction.perVertex

        return mtlVertexDescriptor
    }

    private static func makePipelineDescriptor(layerRenderer: LayerRenderer) throws
        -> MTLRenderPipelineState
    {
        let pipelineDescriptor = Renderer.defaultRenderPipelineDescriptor(
            layerRenderer: layerRenderer)

        let library = layerRenderer.device.makeDefaultLibrary()!

        let vertexFunction = library.makeFunction(name: "tintVertexShader")
        let fragmentFunction = library.makeFunction(name: "tintFragmentShader")

        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.vertexFunction = vertexFunction

        pipelineDescriptor.label = "TriangleRenderPipeline"
        pipelineDescriptor.vertexDescriptor = TintRenderer.buildMetalVertexDescriptor()

        return try layerRenderer.device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }

    func drawCommand(frame: LayerRenderer.Frame) throws -> TintDrawCommand {
        return TintDrawCommand(
            frameIndex: frame.frameIndex,
            uniforms: self.uniformsBuffer[Int(frame.frameIndex % Renderer.maxFramesInFlight)])
    }

    @RendererActor
    func encodeDraw(
        _ drawCommand: TintDrawCommand,
        encoder: MTLRenderCommandEncoder,
        drawable: LayerRenderer.Drawable,
        device: MTLDevice, tintValue: Float,
        buffer: MTLBuffer
    ) {
        encoder.setCullMode(.none)

        encoder.setRenderPipelineState(renderPipelineState)

        var tintUniform: TintUniforms = TintUniforms(tintOpacity: tintValue)
        encoder.setVertexBytes(
            &tintUniform,
            length: MemoryLayout<TintUniforms>.size,
            index: BufferIndex.tintUniforms.rawValue)

        encoder.setVertexBuffer(
            drawCommand.uniforms,
            offset: 0,
            index: BufferIndex.uniforms.rawValue)

        // let bufferLength = MemoryLayout<Vertex>.stride * numVertices

        encoder.setVertexBuffer(
            buffer,
            offset: 0,
            index: BufferIndex.meshPositions.rawValue)
        encoder.drawPrimitives(
            type: .triangle,
            vertexStart: 0,
            vertexCount: numVertices
        )
    }

    @RendererActor
    func updateUniformBuffers(
        _ drawCommand: TintDrawCommand,
        drawable: LayerRenderer.Drawable
    ) {
        drawCommand.uniforms.contents().assumingMemoryBound(to: Uniforms.self).pointee = Uniforms(
            drawable: drawable)
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
    fileprivate let frameIndex: LayerFrameIndex
    fileprivate let uniforms: MTLBuffer & Sendable

    @MainActor
    fileprivate init(frameIndex: LayerFrameIndex, uniforms: MTLBuffer) {
        self.drawCommand = DrawCommand(buffer: uniforms, vertexCount: numVertices)
        self.frameIndex = frameIndex
        self.uniforms = uniforms
    }
}
