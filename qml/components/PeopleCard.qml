pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls
import "../theme" as Room36Theme

Item {
    id: root
    clip: true
    property string title: ""
    property string subtitle: ""
    property string previewSource: ""
    property bool ignored: false
    property bool showIgnore: false
    signal opened()
    signal nameRequested()
    signal ignoreRequested()

    Room36Theme.Theme { id: theme }
    Room36Theme.Typography { id: typography }
    Room36Theme.Metrics { id: metrics }

    Rectangle {
        id: surface
        anchors.fill: parent
        anchors.margins: 7
        radius: metrics.photoCardRadius
        color: theme.panelFill
        border.width: openButton.activeFocus || openButton.hovered ? 2 : 0
        border.color: openButton.activeFocus || openButton.hovered ? theme.focusRing : theme.separator
        clip: true

        Item {
            id: previewFrame
            objectName: "peopleCardPreviewFrame"
            readonly property bool fillsPreview: avatar.fillMode === Image.PreserveAspectCrop
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.top: parent.top
            anchors.bottom: footer.top
            clip: true
            Rectangle { anchors.fill: parent; color: theme.imagePlaceholder }
            Image {
                id: avatar
                objectName: "peopleCardPreviewImage"
                anchors.fill: parent
                source: root.previewSource
                cache: false
                asynchronous: true
                fillMode: Image.PreserveAspectCrop
                clip: true
                Accessible.name: (root.title.length > 0 ? root.title : "Unknown cluster") + " face preview"
            }
        }
        Text {
            anchors.centerIn: previewFrame
            visible: root.previewSource.length === 0 && !root.showIgnore
            text: "Person"
            color: theme.mutedText
            font.pixelSize: typography.body
        }
        Rectangle {
            id: footer
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.bottom: parent.bottom
            height: 56
            radius: surface.radius
            color: theme.footerFill
            Rectangle {
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: parent.top
                height: Math.max(0, parent.height - parent.radius + 1)
                color: theme.footerFill
            }
            Text {
                id: titleText
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: parent.top
                anchors.leftMargin: 12
                anchors.rightMargin: 12
                anchors.topMargin: 5
                text: root.title
                visible: root.title.length > 0
                color: theme.white
                font.family: typography.cardTitleFamily
                font.pixelSize: typography.caption
                elide: Text.ElideRight
            }
            Text {
                anchors.left: parent.left
                anchors.right: root.showIgnore ? nameButton.left : parent.right
                anchors.bottom: parent.bottom
                anchors.bottomMargin: 9
                anchors.leftMargin: 12
                anchors.rightMargin: 4
                text: root.subtitle
                color: theme.white
                font.pixelSize: typography.detail
                elide: Text.ElideRight
            }
            IconButton {
                id: nameButton
                objectName: "personCardName"
                iconName: "edit"
                iconColor: theme.white
                visible: root.showIgnore && !root.ignored
                anchors.right: ignoreButton.left
                anchors.bottom: parent.bottom
                anchors.rightMargin: 3
                anchors.bottomMargin: 6
                buttonSize: 28
                iconSize: 20
                onClicked: root.nameRequested()
                accessibleLabel: "Name " + (root.title.length > 0 ? root.title : "unknown cluster")
            }
            IconButton {
                id: ignoreButton
                objectName: "personCardIgnore"
                iconName: root.ignored ? "arrow_undo" : "dismiss"
                iconColor: theme.white
                visible: root.showIgnore
                anchors.right: parent.right
                anchors.bottom: parent.bottom
                anchors.rightMargin: 8
                anchors.bottomMargin: 6
                buttonSize: 28
                iconSize: 20
                onClicked: root.ignoreRequested()
                accessibleLabel: (root.ignored ? "Restore " : "Ignore ")
                    + (root.title.length > 0 ? root.title : "unknown cluster")
            }
        }
        Button {
            id: openButton
            objectName: "personCardOpen"
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.top: parent.top
            anchors.bottom: footer.top
            opacity: 0.01
            focusPolicy: Qt.StrongFocus
            onClicked: root.opened()
            Accessible.name: "Open " + (root.title.length > 0 ? root.title : "unknown cluster") + ", " + root.subtitle
        }
        RoundedCornerMask {
            anchors.fill: parent
            cornerRadius: surface.radius
            maskColor: theme.libraryBackground
            maskBottomCorners: false
        }
    }
}
