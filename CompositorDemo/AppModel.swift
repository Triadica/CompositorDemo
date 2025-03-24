/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
Shared app state and renderers.
*/

import SwiftUI

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
    public var opacity: Float = 0.8
    var lampsRenderer: LampsRenderer?
}
