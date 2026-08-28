import QtQuick
import "../theme" as Room36Theme

Item {
    id: root

    property real value: 0
    property bool showLabel: true
    property string label: Math.round(normalizedValue * 100) + "%"

    readonly property real normalizedValue: Math.max(0, Math.min(1, value))

    Room36Theme.Theme { id: theme }
    Room36Theme.Typography { id: typography }
    Room36Theme.Metrics { id: metrics }
    Room36Theme.Animations { id: animations }

    implicitWidth: metrics.scoreBarWidth
    implicitHeight: metrics.scoreBarHeight

    Rectangle {
        id: track
        anchors.fill: parent
        radius: metrics.scoreBarRadius
        color: theme.hoverFill
        border.width: 1
        border.color: theme.separator
    }

    Rectangle {
        id: fill
        anchors.left: track.left
        anchors.top: track.top
        anchors.bottom: track.bottom
        width: Math.max(track.height, track.width * root.normalizedValue)
        radius: metrics.scoreBarRadius
        color: root.normalizedValue >= 0.75 ? theme.neonGreen : theme.midBlue

        Behavior on width {
            NumberAnimation {
                duration: animations.selection
                easing.type: animations.standardEasing
            }
        }
    }

    Text {
        anchors.centerIn: parent
        visible: root.showLabel
        text: root.label
        color: root.normalizedValue >= 0.75 ? theme.black : theme.white
        font.family: typography.buttonFamily
        font.pixelSize: typography.caption
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
    }
}
