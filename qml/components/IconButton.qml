import QtQuick
import QtQuick.Controls
import "../theme" as Room36Theme

Button {
    id: root

    property url source: ""
    property string accessibleLabel: text
    property real buttonSize: metrics.iconButtonSize
    property real iconSize: metrics.iconSizeMedium

    Accessible.name: accessibleLabel
    Accessible.role: Accessible.Button

    Room36Theme.Theme { id: theme }
    Room36Theme.Metrics { id: metrics }
    Room36Theme.Animations { id: animations }

    implicitWidth: buttonSize
    implicitHeight: buttonSize
    hoverEnabled: true
    focusPolicy: Qt.StrongFocus

    contentItem: Item {
        Image {
            anchors.centerIn: parent
            width: root.iconSize
            height: root.iconSize
            source: root.source
            fillMode: Image.PreserveAspectFit
            opacity: root.enabled ? 1 : 0.36
        }
    }

    background: Rectangle {
        radius: Math.min(width, height) / 2
        color: !root.enabled
            ? theme.disabledFill
            : root.down
                ? theme.pressedFill
                : root.hovered || root.activeFocus
                    ? theme.hoverFill
                    : Qt.rgba(1, 1, 1, 0)
        border.width: root.activeFocus && root.enabled ? 1 : 0
        border.color: theme.focusRing

        Behavior on color {
            ColorAnimation {
                duration: animations.hover
                easing.type: animations.standardEasing
            }
        }
    }
}
