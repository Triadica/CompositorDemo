/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
The application that creates a scene with a settings view and an immersive interactive view.
*/

import CompositorServices
import SwiftUI

struct ContentStageConfiguration: CompositorLayerConfiguration {
  func makeConfiguration(
    capabilities: LayerRenderer.Capabilities, configuration: inout LayerRenderer.Configuration
  ) {
    configuration.depthFormat = .depth32Float
    configuration.colorFormat = .rgba16Float

    let foveationEnabled = capabilities.supportsFoveation
    configuration.isFoveationEnabled = foveationEnabled

    let options: LayerRenderer.Capabilities.SupportedLayoutsOptions =
      foveationEnabled ? [.foveationEnabled] : []
    let supportedLayouts = capabilities.supportedLayouts(options: options)

    configuration.layout = supportedLayouts.contains(.layered) ? .layered : .dedicated
  }
}

class ResetComputeState: ObservableObject {
  /// every time this value changes, the compute will be reset
  @Published var reset = 1
}

@main
struct InteractionApp: App {

  @State private var appModel = AppModel()

  @StateObject var computeStateNotify = ResetComputeState()

  @Environment(\.dismissImmersiveSpace) var dismissImmersiveSpace
  @Environment(\.openImmersiveSpace) private var openImmersiveSpace

  @StateObject var sharedShaderAddress = SharedShaderAddress()

  var body: some Scene {
    WindowGroup {
      InteractionView()
        .environment(appModel)
        .environmentObject(computeStateNotify)
        .environmentObject(sharedShaderAddress)
        .onAppear {
          let timestamp = Date().formatted(.dateTime.minute().second())
          print("[\(timestamp)] InteractionApp: App appeared")
          if appModel.isFirstLaunch {
            print("[\(timestamp)] InteractionApp: First launch detected, showing immersive space")
            appModel.isFirstLaunch = false
            // Immediately show immersive space on first launch.
            appModel.showImmersiveSpace = true
          }
        }
        .onChange(of: appModel.showImmersiveSpace) { _, newValue in
          // Manage the lifecycle of the immersive space.
          Task { @MainActor in
            let timestamp = Date().formatted(.dateTime.minute().second())
            if newValue {
              print("[\(timestamp)] InteractionApp: Attempting to open immersive space")
              switch await openImmersiveSpace(id: ImmersiveInteractionScene.id) {
              case .opened:
                print("[\(Date().formatted(.dateTime.minute().second()))] InteractionApp: Immersive space opened successfully")
                appModel.immersiveSpaceIsShown = true
              case .error, .userCancelled:
                print("[\(Date().formatted(.dateTime.minute().second()))] InteractionApp: Failed to open immersive space (error or user cancelled)")
                fallthrough
              @unknown default:
                print("[\(Date().formatted(.dateTime.minute().second()))] InteractionApp: Unknown error opening immersive space")
                appModel.immersiveSpaceIsShown = false
                appModel.showImmersiveSpace = false
              }
            } else if appModel.immersiveSpaceIsShown {
              print("[\(timestamp)] InteractionApp: Dismissing immersive space")
              await dismissImmersiveSpace()
              print("[\(Date().formatted(.dateTime.minute().second()))] InteractionApp: Immersive space dismissed")
              appModel.immersiveSpaceIsShown = false
            }
          }
        }
    }
    .windowResizability(.contentSize)
    ImmersiveInteractionScene()
      .environment(appModel)
      .environmentObject(computeStateNotify)
      .environmentObject(sharedShaderAddress)

  }
}
