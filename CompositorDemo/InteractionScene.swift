/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
The main scene for presenting and interacting with content.
*/

import CompositorServices
import SwiftUI

struct ImmersiveInteractionScene: Scene {

    @Environment(AppModel.self) var appModel
    @EnvironmentObject var computeStateNotify: ResetComputeState

    static let id = "ImmersiveInteractionScene"

    var body: some Scene {
        ImmersiveSpace(id: Self.id) {
            CompositorLayer(configuration: ContentStageConfiguration()) { layerRenderer in

                let currentRenderer: CustomRenderer
                do {
                    switch appModel.selectedTab {
                    case .lamps:
                        currentRenderer = try LampsRenderer(layerRenderer: layerRenderer)
                    case .polylines:
                        currentRenderer = try PolylinesRenderer(
                            layerRenderer: layerRenderer
                        )
                    case .triangles:
                        currentRenderer = try TrianglesRenderer(
                            layerRenderer: layerRenderer
                        )
                    case .jsonGen:
                        currentRenderer = try JsonGenRenderer(
                            layerRenderer: layerRenderer
                        )
                    case .attractor:
                        currentRenderer = try AttractorRenderer(
                            layerRenderer: layerRenderer
                        )
                    }
                } catch {
                    fatalError("Failed to create lamps renderer \(error)")
                }

                Task(priority: .high) { @RendererActor in
                    Task { @MainActor in
                        appModel.lampsRenderer = currentRenderer
                    }

                    let renderer = try await Renderer(
                        layerRenderer,
                        appModel,
                        currentRenderer)
                    try await renderer.renderLoop()

                    Task { @MainActor in
                        appModel.lampsRenderer = nil
                    }
                }
                layerRenderer.onSpatialEvent = {
                  currentRenderer.onSpatialEvents(events: $0)
                }

            }
        }
        .immersionStyle(selection: .constant(appModel.immersionStyle), in: .mixed, .full)
        .upperLimbVisibility(appModel.upperLimbVisibility)
        .onChange(of: computeStateNotify.reset) { oldValue, newValue in
            appModel.lampsRenderer?.resetComputeState()
        }
    }
}
