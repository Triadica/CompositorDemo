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

                let lampsRenderer: CustomRenderer
                do {
                    switch appModel.selectedTab {
                    case .lamps:
                        print("Lamps selected")
                        lampsRenderer = try LampsRenderer(layerRenderer: layerRenderer)
                    case .polylines:
                        print("Polylines selected")
                        lampsRenderer = try PolylinesRenderer(
                            layerRenderer: layerRenderer
                        )
                    }
                } catch {
                    fatalError("Failed to create lamps renderer \(error)")
                }

                Task(priority: .high) { @RendererActor in
                    Task { @MainActor in
                        appModel.lampsRenderer = lampsRenderer
                    }

                    let renderer = try await Renderer(
                        layerRenderer,
                        appModel,
                        lampsRenderer)
                    try await renderer.renderLoop()

                    Task { @MainActor in
                        appModel.lampsRenderer = nil
                    }
                }
                layerRenderer.onSpatialEvent = {
                    lampsRenderer.onSpatialEvents(events: $0)
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
