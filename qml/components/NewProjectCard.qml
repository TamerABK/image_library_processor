import QtQuick
import QtQuick.Layouts
import "../theme" as Room36Theme

Item {
    id: root
    property string viewSize: "Medium"

    signal clicked()

    Room36Theme.Theme { id: theme }
    Room36Theme.Typography { id: typography }
    Room36Theme.Metrics { id: metrics }
    Room36Theme.Animations { id: animations }

    implicitWidth: metrics.projectCardReferenceWidth
    implicitHeight: metrics.projectCardReferenceHeight
    opacity: enabled ? 1 : 0.4

    HoverHandler { id: hoverHandler }
    TapHandler { onTapped: root.clicked() }

    Room36Theme.Effects {
        z: -1
        sourceItem: surface
        active: hoverHandler.hovered
        glowColor: theme.selectedGlow
    }

    Rectangle {
        id: surface
        anchors.fill: parent
        radius: metrics.projectCardRadius
        color: hoverHandler.hovered ? theme.selectedFill : theme.lightBlue

        Behavior on color {
            ColorAnimation {
                duration: animations.hover
                easing.type: animations.standardEasing
            }
        }

        ColumnLayout {
            anchors.centerIn: parent
            width: parent.width * 0.70
            spacing: metrics.spacingSmall

            FluentIcon {
                Layout.alignment: Qt.AlignHCenter
                Layout.preferredWidth: root.viewSize === "Small" ? 46 : root.viewSize === "Large" ? 70 : 58
                Layout.preferredHeight: Layout.preferredWidth
                name: "arrow_download"
                size: 48
                color: theme.white
            }

            Text {
                Layout.fillWidth: true
                text: "New project"
                color: theme.white
                font.family: typography.bodyFamily
                font.pixelSize: root.viewSize === "Small" ? typography.body
                    : root.viewSize === "Large" ? typography.sectionTitle + 2 : typography.sectionTitle
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                elide: Text.ElideRight
            }
        }
    }
}
