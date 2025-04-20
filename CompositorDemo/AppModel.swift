/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
Shared app state and renderers.
*/

import SwiftUI

enum DemoTab: String, CaseIterable, Identifiable {
    case lamps
    case polylines
    case triangles
    case jsonGen
    case attractor
    case blocks

    var id: Self { self }
}

/// Maintains app-wide state.
@Observable
public class AppModel {
    // App state
    public var isFirstLaunch = true
    public var showImmersiveSpace = false
    public var immersiveSpaceIsShown = false
    public var immersionStyle: ImmersionStyle = .full

    // Limb visibility
    public var upperLimbVisibility: Visibility = .visible

    // Content rendering
    public var opacity: Float = 1.0

    var selectedTab: DemoTab = .blocks
    var lampsRenderer: CustomRenderer?
}
