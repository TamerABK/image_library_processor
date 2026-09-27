import QtQuick
import QtQuick.Controls
import "../theme" as Room36Theme

Button {
    id: root

    property url iconSource: ""
    property string provider: ""
    property string accessibleLabel: provider.length > 0 ? provider : text

    Accessible.name: accessibleLabel
    Accessible.role: Accessible.Button

    Room36Theme.Theme { id: theme }
    Room36Theme.Metrics { id: metrics }
    Room36Theme.Animations { id: animations }

    implicitWidth: metrics.authSocialButtonSize
    implicitHeight: metrics.authSocialButtonSize
    hoverEnabled: true
    focusPolicy: Qt.StrongFocus

    contentItem: Item {
        Image {
            anchors.centerIn: parent
            width: metrics.iconSizeLarge
            height: metrics.iconSizeLarge
            source: root.iconSource
            cache: true
            fillMode: Image.PreserveAspectFit
            opacity: root.enabled ? 1 : 0.36
        }
    }

    background: Rectangle {
        radius: metrics.authSocialButtonRadius
        color: !root.enabled
            ? theme.disabledFill
            : root.down
                ? theme.midBlue
                : root.hovered || root.activeFocus
                    ? theme.midBlue
                    : theme.mainBlue
        border.width: root.activeFocus && root.enabled ? 2 : 0
        border.color: theme.focusRing

        Behavior on color {
            ColorAnimation {
                duration: animations.hover
                easing.type: animations.standardEasing
            }
        }
    }
}
