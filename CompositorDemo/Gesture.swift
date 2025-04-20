/// implement basic moving and scaling gesture, however not simply following your finger.
/// it will still perform operation when you finger finished moving since it changes velocity.

import RealityKit
import Spatial
import SwiftUI
import simd

class GestureManager {
  /// update this with gesture events
  var pinchStart: (SIMD3<Float>, Chirality)? = nil
  var viewerPosition: SIMD3<Float> = SIMD3<Float>(0, 0, 0)

  /// initial length when the other chirality pinch started
  var pinchBaseLength: Float = 0.0
  var viewerScale: Float = 1.0
  var scaleStartedBy: Chirality? = nil

  /// track the position pinch started, following pinches define the velocity of moving, to update self.viewerPosition .
  /// other other chirality events are used for scaling the entity
  func onSpatialEvent(event: SpatialEventCollection.Event) {
    guard let chirality = event.chirality,
      event.inputDevicePose?.pose3D != nil,
      event.inputDevicePose?.pose3D.position != nil,
      event.inputDevicePose?.pose3D.rotation != nil
    else {
      return
    }

    if pinchStart == nil {
      handlePinchStart(event: event, chirality: chirality)
    } else {
      handlePinchActive(event: event, chirality: chirality)
    }

    if event.phase == .ended {
      self.scaleStartedBy = nil
    }
  }

  private func handlePinchStart(event: SpatialEventCollection.Event, chirality: Chirality) {
    if event.phase == .active {
      if self.scaleStartedBy != nil && self.scaleStartedBy == chirality {
        // nothing
      } else {
        pinchStart = (
          event.inputDevicePose!.pose3D.position.to_simd3,
          chirality
        )
      }
    }
  }

  private func handlePinchActive(event: SpatialEventCollection.Event, chirality: Chirality) {
    guard let pinchStart = self.pinchStart else {
      return
    }

    if event.phase == .ended {
      if event.chirality == pinchStart.1 {
        self.pinchStart = nil
      } else {
        self.scaleStartedBy = nil
        self.pinchStart = nil
      }
    } else if event.phase == .active {
      guard let pinchPosition = event.inputDevicePose?.pose3D.position.to_simd3 else {
        return
      }

      let startPosition = pinchStart.0
      let pinchDelta = simd_distance(pinchPosition, startPosition)
      if event.chirality == pinchStart.1 {
        if scaleStartedBy == nil {
          // update the viewer position
          self.viewerPosition -= (pinchPosition - startPosition) * 0.1
        }
      } else {
        if scaleStartedBy == nil {
          pinchBaseLength = pinchDelta
          scaleStartedBy = event.chirality
        } else {
          let ratio = pow(pinchDelta / pinchBaseLength, 0.2)
          self.viewerScale *= ratio
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
