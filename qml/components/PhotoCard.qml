pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Layouts
import "../theme" as Room36Theme

Item {
    id: root

    property url source: ""
    property string filename: ""
    property bool selected: false
    property bool favorite: false
    property bool showCheckbox: true
    property bool showFavorite: true
    property bool showMenu: true
    property string badgeText: ""
    readonly property bool hasImageSource: String(source).length > 0

    signal clicked()
    signal selectionToggled(bool selected)
    signal favoriteToggled(bool favorite)
    signal menuRequested()

    Room36Theme.Theme { id: theme }
    Room36Theme.Typography { id: typography }
    Room36Theme.Metrics { id: metrics }
    Room36Theme.Animations { id: animations }

    implicitWidth: metrics.photoCardReferenceWidth
    implicitHeight: metrics.photoCardReferenceHeight

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
        radius: metrics.photoCardRadius
        color: theme.panelFill
        border.width: root.selected ? 2 : 1
        border.color: root.selected ? theme.focusRing : theme.separator
        clip: true

        Rectangle {
            id: imageViewport
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.top: parent.top
            height: parent.height - metrics.photoCardFooterHeight
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
                source: theme.iconUrl("landscape.svg")
                fillMode: Image.PreserveAspectFit
                visible: !root.hasImageSource
                opacity: 0.62
            }

            Rectangle {
                anchors.left: parent.left
                anchors.top: parent.top
                anchors.leftMargin: metrics.spacingSmall
                anchors.topMargin: metrics.spacingSmall
                height: 24
                width: Math.max(48, badgeLabel.implicitWidth + metrics.spacingSmall)
                radius: height / 2
                color: theme.mainBlue
                visible: root.badgeText.length > 0

                Text {
                    id: badgeLabel
                    anchors.centerIn: parent
                    text: root.badgeText
                    color: theme.white
                    font.family: typography.buttonFamily
                    font.pixelSize: typography.detail
                    elide: Text.ElideRight
                }
            }

            SelectionCheckbox {
                anchors.left: parent.left
                anchors.bottom: parent.bottom
                anchors.leftMargin: metrics.spacingSmall
                anchors.bottomMargin: metrics.spacingSmall
                checked: root.selected
                visible: root.showCheckbox
                onToggled: root.selectionToggled(checked)
            }

            IconButton {
                anchors.right: parent.right
                anchors.top: parent.top
                anchors.rightMargin: metrics.spacingTiny
                anchors.topMargin: metrics.spacingTiny
                source: theme.iconUrl(root.favorite ? "favorite_filled.svg" : "favorite.svg")
                accessibleLabel: "Favorite"
                buttonSize: 32
                iconSize: metrics.iconSizeMedium
                visible: root.showFavorite
                onClicked: root.favoriteToggled(!root.favorite)
            }
        }

        Rectangle {
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.bottom: parent.bottom
            height: metrics.photoCardFooterHeight
            color: theme.footerFill

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: metrics.spacingSmall
                anchors.rightMargin: metrics.spacingTiny
                spacing: metrics.spacingTiny

                Text {
                    Layout.fillWidth: true
                    text: root.filename
                    color: theme.white
                    font.family: typography.cardTitleFamily
                    font.pixelSize: typography.caption
                    verticalAlignment: Text.AlignVCenter
                    elide: Text.ElideMiddle
                }

                IconButton {
                    source: theme.iconUrl("menu_more.svg")
                    accessibleLabel: "Photo menu"
                    buttonSize: 30
                    iconSize: metrics.iconSizeMedium
                    visible: root.showMenu
                    onClicked: root.menuRequested()
                }
            }
        }
    }
}
