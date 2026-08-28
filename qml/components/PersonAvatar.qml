import QtQuick
import "../theme" as Room36Theme

Item {
    id: root

    property url source: ""
    property string name: ""
    property int count: -1
    property bool showName: false
    property bool showCount: count >= 0
    property bool selected: false

    signal clicked()

    Room36Theme.Theme { id: theme }
    Room36Theme.Typography { id: typography }
    Room36Theme.Metrics { id: metrics }
    Room36Theme.Animations { id: animations }

    implicitWidth: metrics.personAvatarSize
    implicitHeight: metrics.personAvatarSize

    HoverHandler { id: hoverHandler }
    TapHandler { onTapped: root.clicked() }

    Room36Theme.Effects {
        z: -1
        sourceItem: avatarSurface
        active: root.selected || hoverHandler.hovered
        glowColor: theme.selectedGlow
    }

    Rectangle {
        id: avatarSurface
        anchors.fill: parent
        radius: metrics.personAvatarRadius
        color: theme.imagePlaceholder
        border.width: root.selected ? 2 : 1
        border.color: root.selected ? theme.focusRing : theme.separator
        clip: true

        Image {
            id: avatarImage
            anchors.fill: parent
            source: root.source
            fillMode: Image.PreserveAspectCrop
            visible: String(root.source).length > 0 && status !== Image.Error
        }

        Image {
            anchors.centerIn: parent
            width: metrics.iconSizeLarge
            height: metrics.iconSizeLarge
            source: theme.iconUrl("profile.svg")
            fillMode: Image.PreserveAspectFit
            visible: !avatarImage.visible
            opacity: 0.62
        }

        Rectangle {
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.bottom: parent.bottom
            height: 24
            visible: root.showName && root.name.length > 0
            color: theme.overlayScrim

            Text {
                anchors.fill: parent
                anchors.leftMargin: metrics.spacingTiny
                anchors.rightMargin: metrics.spacingTiny
                text: root.name
                color: theme.white
                font.family: typography.captionFamily
                font.pixelSize: typography.detail
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                elide: Text.ElideRight
            }
        }
    }

    Rectangle {
        anchors.right: parent.right
        anchors.top: parent.top
        anchors.rightMargin: -metrics.spacingTiny
        anchors.topMargin: -metrics.spacingTiny
        width: Math.max(24, countLabel.implicitWidth + metrics.spacingTiny * 2)
        height: 22
        radius: height / 2
        color: theme.mainBlue
        visible: root.showCount

        Text {
            id: countLabel
            anchors.centerIn: parent
            text: root.count
            color: theme.white
            font.family: typography.buttonFamily
            font.pixelSize: typography.detail
        }
    }
}
