pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Layouts
import "../theme" as Room36Theme

Item {
    id: root

    property url source: ""
    property string title: ""
    property string subtitle: ""
    property int photoCount: -1
    property bool selected: false
    property bool menuEnabled: true
    property color cornerMaskColor: theme.homeBackground
    readonly property bool hasImageSource: String(source).length > 0

    signal clicked()
    signal menuRequested()

    Room36Theme.Theme { id: theme }
    Room36Theme.Typography { id: typography }
    Room36Theme.Metrics { id: metrics }
    Room36Theme.Animations { id: animations }

    implicitWidth: metrics.projectCardReferenceWidth
    implicitHeight: metrics.projectCardReferenceHeight

    HoverHandler { id: hoverHandler }
    TapHandler { onTapped: root.clicked() }

    Room36Theme.Effects {
        z: -1
        sourceItem: cardSurface
        active: root.selected || hoverHandler.hovered
        glowColor: theme.selectedGlow
    }

    Rectangle {
        id: cardSurface
        anchors.fill: parent
        radius: metrics.projectCardRadius
        color: theme.panelFill
        border.width: root.selected ? 2 : 1
        border.color: root.selected ? theme.focusRing : theme.separator
        clip: true

        Rectangle {
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.top: parent.top
            height: metrics.projectCardImageHeight
            color: theme.imagePlaceholder

            Loader {
                id: previewLoader
                anchors.fill: parent
                active: root.hasImageSource

                sourceComponent: Image {
                    anchors.fill: parent
                    source: root.source
                    cache: true
                    fillMode: Image.PreserveAspectCrop
                }
            }

            Image {
                anchors.centerIn: parent
                width: metrics.iconSizeLarge
                height: metrics.iconSizeLarge
                source: theme.iconUrl("project_open.svg")
                fillMode: Image.PreserveAspectFit
                visible: !root.hasImageSource
                opacity: 0.62
            }

            IconButton {
                objectName: "projectMenuButton"
                enabled: root.menuEnabled
                anchors.right: parent.right
                anchors.top: parent.top
                anchors.rightMargin: metrics.spacingTiny
                anchors.topMargin: metrics.spacingTiny
                source: theme.iconUrl("menu_more.svg")
                accessibleLabel: "Project menu"
                buttonSize: 32
                iconSize: metrics.iconSizeMedium
                onClicked: root.menuRequested()
            }
        }

        Rectangle {
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.bottom: parent.bottom
            height: metrics.projectCardFooterHeight
            color: theme.footerFill

            ColumnLayout {
                anchors.fill: parent
                anchors.leftMargin: metrics.spacingSmall
                anchors.rightMargin: metrics.spacingSmall
                spacing: 0

                Text {
                    Layout.fillWidth: true
                    text: root.title
                    color: theme.white
                    font.family: typography.cardTitleFamily
                    font.pixelSize: typography.body
                    elide: Text.ElideRight
                    verticalAlignment: Text.AlignVCenter
                }

                Text {
                    Layout.fillWidth: true
                    text: root.subtitle.length > 0
                        ? root.subtitle
                        : (root.photoCount >= 0 ? root.photoCount + " photos" : "")
                    visible: text.length > 0
                    color: theme.lightBlue
                    font.family: typography.detailFamily
                    font.pixelSize: typography.caption
                    elide: Text.ElideRight
                    verticalAlignment: Text.AlignVCenter
                }
            }
        }

        RoundedCornerMask {
            anchors.fill: parent
            cornerRadius: metrics.projectCardRadius
            maskColor: root.cornerMaskColor
        }
    }
}
