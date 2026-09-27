import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../theme" as Room36Theme

Button {
    id: root

    property url iconSource: ""
    property bool busy: false
    property real radius: height / 2

    Room36Theme.Theme { id: theme }
    Room36Theme.Typography { id: typography }
    Room36Theme.Metrics { id: metrics }
    Room36Theme.Animations { id: animations }

    implicitWidth: Math.max(132, contentLayout.implicitWidth + leftPadding + rightPadding)
    implicitHeight: metrics.toolbarButtonHeight
    leftPadding: metrics.spacingMedium
    rightPadding: metrics.spacingMedium
    topPadding: 0
    bottomPadding: 0
    hoverEnabled: true
    focusPolicy: Qt.StrongFocus

    contentItem: RowLayout {
        id: contentLayout

        spacing: metrics.spacingSmall

        BusyIndicator {
            Layout.preferredWidth: metrics.iconSizeMedium
            Layout.preferredHeight: metrics.iconSizeMedium
            running: root.busy
            visible: root.busy
        }

        Image {
            Layout.preferredWidth: metrics.iconSizeMedium
            Layout.preferredHeight: metrics.iconSizeMedium
            source: root.iconSource
            visible: !root.busy && String(root.iconSource).length > 0
            fillMode: Image.PreserveAspectFit
            opacity: root.enabled ? 1 : 0.38
        }

        Text {
            id: label
            Layout.fillWidth: true
            text: root.text
            color: root.enabled ? theme.white : theme.disabledText
            font.family: typography.buttonFamily
            font.pixelSize: typography.button
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            elide: Text.ElideRight
        }
    }

    background: Rectangle {
        radius: root.radius
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
