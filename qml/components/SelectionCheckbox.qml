import QtQuick
import QtQuick.Controls
import "../theme" as Room36Theme

CheckBox {
    id: root

    Room36Theme.Theme { id: theme }
    Room36Theme.Typography { id: typography }
    Room36Theme.Metrics { id: metrics }
    Room36Theme.Animations { id: animations }

    implicitWidth: indicatorItem.implicitWidth + (text.length > 0 ? label.implicitWidth + spacing : 0)
    implicitHeight: Math.max(indicatorItem.implicitHeight, label.implicitHeight)
    spacing: metrics.spacingTiny
    hoverEnabled: true
    focusPolicy: Qt.StrongFocus

    indicator: Rectangle {
        id: indicatorItem
        implicitWidth: metrics.sidebarCheckboxSize
        implicitHeight: metrics.sidebarCheckboxSize
        x: root.leftPadding
        y: (root.height - height) / 2
        radius: 2
        color: root.checked ? theme.mainBlue : theme.panelRaisedFill
        border.width: root.activeFocus ? 2 : 1
        border.color: root.activeFocus ? theme.focusRing : theme.mainBlue

        Rectangle {
            anchors.centerIn: parent
            width: parent.width * 0.48
            height: parent.height * 0.48
            radius: 1
            visible: root.checked
            color: theme.neonGreen
        }

        Behavior on color {
            ColorAnimation {
                duration: animations.selection
                easing.type: animations.standardEasing
            }
        }
    }

    contentItem: Text {
        id: label
        text: root.text
        visible: root.text.length > 0
        leftPadding: root.indicator.width + root.spacing
        color: root.enabled ? theme.black : theme.disabledText
        font.family: typography.captionFamily
        font.pixelSize: typography.caption
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }
}
