import QtQuick
import QtQuick.Controls
import "../theme" as Room36Theme

Button {
    id: root

    property url source: ""
    property string iconName: ""
    property string iconVariant: "regular"
    property color iconColor: theme.black
    property string accessibleLabel: text
    property real buttonSize: metrics.iconButtonSize
    property real iconSize: metrics.iconSizeMedium
    property real radius: Math.min(width, height) / 2

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
        FluentIcon {
            anchors.centerIn: parent
            width: root.iconSize
            height: root.iconSize
            name: root.iconName
            size: root.iconSize <= 17 ? 16 : root.iconSize <= 22 ? 20 : 24
            variant: root.iconVariant
            color: root.iconColor
            hovered: root.hovered
            enabled: root.enabled
            visible: root.iconName.length > 0
        }
        Image {
            anchors.centerIn: parent
            width: root.iconSize
            height: root.iconSize
            source: root.source
            visible: root.iconName.length === 0
            fillMode: Image.PreserveAspectFit
            opacity: root.enabled ? 1 : 0.36
        }
    }

    background: Rectangle {
        radius: root.radius
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
