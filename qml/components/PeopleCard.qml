pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls
import "../theme" as Room36Theme

Item {
    id: root
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
        border.width: openButton.activeFocus || openButton.hovered ? 2 : 1
        border.color: openButton.activeFocus || openButton.hovered ? theme.focusRing : theme.separator
        clip: true

        Rectangle { anchors.fill: avatar; color: theme.imagePlaceholder }
        Image {
            id: avatar
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.top: parent.top
            anchors.bottom: footer.top
            source: root.previewSource
            cache: false
            asynchronous: true
            fillMode: Image.PreserveAspectCrop
            Accessible.name: (root.title.length > 0 ? root.title : "Unknown cluster") + " face preview"
        }
        Text {
            anchors.centerIn: avatar
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
            height: root.showIgnore ? 68 : 56
            color: theme.footerFill
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
                anchors.right: parent.right
                anchors.top: titleText.visible ? titleText.bottom : parent.top
                anchors.topMargin: titleText.visible ? 0 : 7
                anchors.leftMargin: 12
                anchors.rightMargin: 12
                text: root.subtitle
                color: theme.white
                font.pixelSize: typography.detail
                elide: Text.ElideRight
            }
            PillButton {
                id: nameButton
                objectName: "personCardName"
                variant: "filled"
                compact: true
                vignette: true
                visible: root.showIgnore && !root.ignored
                anchors.left: parent.left
                anchors.bottom: parent.bottom
                anchors.leftMargin: 8
                anchors.bottomMargin: 10
                height: 26
                text: "Name"
                onClicked: root.nameRequested()
                Accessible.name: "Name " + (root.title.length > 0 ? root.title : "unknown cluster")
            }
            PillButton {
                id: ignoreButton
                objectName: "personCardIgnore"
                variant: "filled"
                compact: true
                vignette: true
                visible: root.showIgnore
                anchors.right: parent.right
                anchors.bottom: parent.bottom
                anchors.rightMargin: 8
                anchors.bottomMargin: 10
                height: 26
                text: root.ignored ? "Restore" : "Ignore"
                onClicked: root.ignoreRequested()
                Accessible.name: (root.ignored ? "Restore " : "Ignore ")
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
    }
}
