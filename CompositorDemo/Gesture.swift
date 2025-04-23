/// implement basic moving and scaling gesture, however not simply following your finger.
/// it will still perform operation when you finger finished moving since it changes velocity.

import RealityKit
import Spatial
import SwiftUI
import simd

private struct PinchHappen {
  var position: SIMD3<Float>
  var chirality: Chirality
}

class GestureManager {
  /// update this with gesture events
  private var primaryStarted: PinchHappen? = nil
  var secondaryStarted: Chirality? = nil
  var secondaryStartPosition: SIMD3<Float> = SIMD3<Float>(0, 0, 0)

  var viewerPosition: SIMD3<Float> = SIMD3<Float>(0, 0, 0)
  var viewerScale: Float = 1.0
  /// rotation of the viewer, in radians
  var viewerRotation: Float = 0.0

  /// initial length when the other chirality pinch started
  var pinchBaseLength: Float = 0.0
  /// initial angle when the other chirality pinch started
  var pinchBaseRadian: Float = 0.0

  var onScene: Bool = false
  var gestureDirection: Float {
    if onScene {
      return -1.0
    } else {
      return 1.0
    }
  }

  init(onScene: Bool = false) {
    self.onScene = onScene
  }

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

    if primaryStarted == nil {
      handlePinchStart(event: event, chirality: chirality)
    } else {
      handlePinchActive(event: event, chirality: chirality)
    }

    if event.phase == .ended {
      self.secondaryStarted = nil
    }
  }

  private func handlePinchStart(event: SpatialEventCollection.Event, chirality: Chirality) {
    if event.phase == .active {
      if self.secondaryStarted != nil && self.secondaryStarted == chirality {
        // nothing
      } else {
        primaryStarted = PinchHappen(
          position: event.inputDevicePose!.pose3D.position.to_simd3,
          chirality: chirality
        )
      }
    }
  }

  private func handlePinchActive(event: SpatialEventCollection.Event, chirality: Chirality) {
    guard let pinchStart = self.primaryStarted else {
      return
    }

    if event.phase == .ended {
      if event.chirality == pinchStart.chirality {
        self.primaryStarted = nil
      } else {
        self.secondaryStarted = nil
        self.primaryStarted = nil
      }
    } else if event.phase == .active {
      guard let pinchPosition = event.inputDevicePose?.pose3D.position.to_simd3 else {
        return
      }

      let startPosition: SIMD3<Float> = pinchStart.position
      if event.chirality == pinchStart.chirality {
        if secondaryStarted == nil {
          // update the viewer position
          var delta = pinchPosition - startPosition

          if self.gestureDirection < 0 {
            // on scene, we need rotate the delta vector
            let rotation: Float = -viewerRotation
            let cosRadian = cos(rotation)
            let sinRadian = sin(rotation)
            /// make new delta since we rotate the world viewer
            delta = SIMD3<Float>(
              delta.x * cosRadian - delta.z * sinRadian,
              delta.y,
              delta.x * sinRadian + delta.z * cosRadian
            )
          }
          self.viewerPosition -= delta * 0.1 * gestureDirection
        }
      } else {
        let pinchDelta = simd_distance(pinchPosition, startPosition)
        let pinchRadian = atan2(
          pinchPosition.z - startPosition.z, pinchPosition.x - startPosition.x)
        if secondaryStarted == nil {
          pinchBaseLength = pinchDelta
          pinchBaseRadian = pinchRadian
          secondaryStarted = event.chirality
          secondaryStartPosition = pinchPosition

        } else {

          let pinchAt2 = SIMD2(pinchPosition.x, pinchPosition.z)
          let startAt2 = SIMD2(startPosition.x, startPosition.z)
          let secondaryStart2 = SIMD2(secondaryStartPosition.x, secondaryStartPosition.z)

          let secondaryDirection = simd_normalize(pinchAt2 - secondaryStart2)
          let secondaryArmDirection = simd_normalize(startAt2 - secondaryStart2)
          let guessScaleOrRotate = abs(simd_dot(secondaryDirection, secondaryArmDirection))

          if guessScaleOrRotate > 0.8 {

            let ratio: Float = pow(pinchDelta / pinchBaseLength, 0.2)
            self.viewerScale *= ratio
          } else if guessScaleOrRotate < 0.7 {

            var deltaRadian = pinchRadian - pinchBaseRadian
            if deltaRadian > .pi {
              deltaRadian -= 2 * .pi
            } else if deltaRadian < -.pi {
              deltaRadian += 2 * .pi
            }
            self.viewerRotation += deltaRadian * 0.02 * gestureDirection
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
