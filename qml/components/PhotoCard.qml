pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Layouts
import "../theme" as Room36Theme

Item {
    id: root

    property url source: ""
    property bool cacheImage: true
    property string filename: ""
    property bool selected: false
    property bool favorite: false
    property bool favoriteBusy: false
    property bool showCheckbox: true
    property bool showFavorite: true
    property bool showMenu: true
    property string badgeText: ""
    property string warningLevel: ""
    property color cornerMaskColor: theme.libraryBackground
    readonly property bool hasImageSource: String(source).length > 0

    signal clicked()
    signal imageClicked()
    signal footerClicked()
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
        border.width: root.selected ? 2 : 0
        border.color: root.selected ? theme.focusRing : theme.separator
        clip: true

        Rectangle {
            id: imageViewport
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.top: parent.top
            height: parent.height - metrics.photoCardFooterHeight
            color: theme.imagePlaceholder

            MouseArea {
                objectName: "photoCardImage"
                anchors.fill: parent
                onClicked: root.imageClicked()
            }

            Loader {
                id: previewLoader
                anchors.fill: parent
                active: root.hasImageSource

                sourceComponent: Image {
                    anchors.fill: parent
                    source: root.source
                    cache: root.cacheImage
                    asynchronous: true
                    fillMode: Image.PreserveAspectCrop
                }
            }

            FluentIcon {
                anchors.centerIn: parent
                width: metrics.iconSizeLarge
                height: metrics.iconSizeLarge
                name: "image"
                size: 24
                color: theme.midBlue
                visible: !root.hasImageSource
                opacity: 0.62
            }

            Rectangle {
                objectName: "photoCardBadge"
                anchors.left: parent.left
                anchors.top: parent.top
                anchors.leftMargin: metrics.spacingSmall
                anchors.topMargin: metrics.spacingSmall
                height: 24
                width: Math.max(48, badgeLabel.implicitWidth + metrics.spacingSmall)
                radius: height / 2
                color: theme.mainBlue
                visible: root.badgeText.length > 0 && root.warningLevel.length === 0

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

            FluentIcon {
                objectName: "photoWarningIcon"
                anchors.left: parent.left
                anchors.top: parent.top
                anchors.leftMargin: metrics.spacingSmall
                anchors.topMargin: metrics.spacingSmall
                name: "warning"
                size: 20
                color: root.warningLevel === "red" ? theme.warningRed : theme.warningYellow
                visible: root.warningLevel.length > 0
                Accessible.name: root.warningLevel === "red" ? "Blurry" : "Needs review"
            }

            SelectionCheckbox {
                objectName: "photoSelectionCheckbox"
                anchors.right: parent.right
                anchors.top: parent.top
                anchors.rightMargin: metrics.spacingSmall
                anchors.topMargin: metrics.spacingSmall
                checked: root.selected
                visible: root.showCheckbox
                onToggled: root.selectionToggled(checked)
            }

        }

        Rectangle {
            id: footer
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.bottom: parent.bottom
            height: metrics.photoCardFooterHeight
            radius: cardSurface.radius
            color: theme.footerFill

            Rectangle {
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: parent.top
                height: Math.max(0, parent.height - parent.radius + 1)
                color: theme.footerFill
            }

            MouseArea {
                objectName: "photoCardFooter"
                anchors.fill: parent
                onClicked: root.footerClicked()
            }

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
                    iconName: "more_horizontal"
                    iconColor: theme.white
                    accessibleLabel: "Photo menu"
                    buttonSize: 30
                    iconSize: metrics.iconSizeMedium
                    visible: root.showMenu
                    onClicked: root.menuRequested()
                }
                IconButton {
                    objectName: "photoFavoriteButton"
                    iconName: "star"
                    iconVariant: root.favorite ? "filled" : "regular"
                    iconColor: theme.white
                    accessibleLabel: root.favorite ? "Remove favorite" : "Add favorite"
                    buttonSize: 30
                    iconSize: metrics.iconSizeMedium
                    visible: root.showFavorite
                    enabled: !root.favoriteBusy
                    onClicked: root.favoriteToggled(!root.favorite)
                }
            }
        }
        RoundedCornerMask {
            anchors.fill: parent
            cornerRadius: cardSurface.radius
            maskColor: root.cornerMaskColor
            maskBottomCorners: false
        }
    }
}
