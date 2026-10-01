import QtQuick
import QtQuick.Layouts
import "../theme" as Room36Theme

Item {
    id: root

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

            Image {
                Layout.alignment: Qt.AlignHCenter
                Layout.preferredWidth: 58
                Layout.preferredHeight: 58
                source: theme.iconUrl("import_home.svg")
                cache: true
                fillMode: Image.PreserveAspectFit
            }

            Text {
                Layout.fillWidth: true
                text: "New project"
                color: theme.white
                font.family: typography.bodyFamily
                font.pixelSize: typography.sectionTitle
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                elide: Text.ElideRight
            }
        }
    }
}
