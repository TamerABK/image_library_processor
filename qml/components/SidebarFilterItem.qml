import QtQuick
import QtQuick.Controls
import QtQuick.Effects
import QtQuick.Layouts
import "../theme" as Room36Theme

Button {
    id: root

    property url iconSource: ""
    property string label: text
    property int count: -1
    property bool selected: false
    property bool expanded: false
    property bool showChevron: false
    property bool darkSurface: false

    Accessible.name: count >= 0 ? label + ", " + count : label
    Accessible.role: Accessible.Button
    Accessible.checkable: true
    Accessible.checked: selected

    Room36Theme.Theme { id: theme }
    Room36Theme.Typography { id: typography }
    Room36Theme.Metrics { id: metrics }
    Room36Theme.Animations { id: animations }

    implicitWidth: metrics.sidebarNavRowWidth
    implicitHeight: metrics.sidebarNavRowHeight
    leftPadding: metrics.spacingSmall
    rightPadding: metrics.spacingSmall
    hoverEnabled: true
    focusPolicy: Qt.StrongFocus

    contentItem: RowLayout {
        spacing: metrics.spacingSmall

        Image {
            Layout.preferredWidth: metrics.iconSizeMedium
            Layout.preferredHeight: metrics.iconSizeMedium
            source: root.iconSource
            visible: String(root.iconSource).length > 0
            fillMode: Image.PreserveAspectFit
            opacity: root.enabled ? 1 : 0.36
            layer.enabled: root.darkSurface
            layer.effect: MultiEffect {
                colorization: 1
                colorizationColor: theme.white
            }
        }

        Text {
            Layout.fillWidth: true
            text: root.label
            color: root.enabled ? (root.darkSurface ? theme.white : theme.black) : theme.disabledText
            font.family: root.selected ? typography.buttonFamily : typography.bodyFamily
            font.pixelSize: typography.caption
            elide: Text.ElideRight
            verticalAlignment: Text.AlignVCenter
        }

        Text {
            text: root.count >= 0 ? root.count : ""
            visible: root.count >= 0
            color: root.enabled ? (root.darkSurface ? theme.white : theme.mutedText) : theme.disabledText
            font.family: typography.detailFamily
            font.pixelSize: typography.detail
            verticalAlignment: Text.AlignVCenter
        }

        Image {
            Layout.preferredWidth: metrics.iconSizeSmall
            Layout.preferredHeight: metrics.iconSizeSmall
            source: theme.iconUrl(root.expanded ? "chevron_down.svg" : "chevron_right.svg")
            visible: root.showChevron
            fillMode: Image.PreserveAspectFit
            opacity: root.enabled ? 1 : 0.36
        }
    }

    background: Item {
        Rectangle {
            id: surface
            anchors.fill: parent
            radius: metrics.sidebarNavRowRadius
            color: !root.enabled
                ? theme.disabledFill
                : root.down
                    ? theme.pressedFill
                    : root.selected
                        ? theme.selectedFill
                        : root.hovered || root.activeFocus
                            ? theme.hoverFill
                            : (root.darkSurface ? theme.libraryNavFill : Qt.rgba(1, 1, 1, 0))
            border.width: root.activeFocus && root.enabled ? 1 : 0
            border.color: theme.focusRing

            Behavior on color {
                ColorAnimation {
                    duration: animations.hover
                    easing.type: animations.standardEasing
                }
            }
        }

        Room36Theme.Effects {
            z: -1
            sourceItem: surface
            active: root.enabled && (root.selected || root.activeFocus)
            glowColor: theme.selectedGlow
        }
    }
}
