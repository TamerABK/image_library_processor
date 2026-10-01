import QtQuick
import QtQuick.Layouts
import "../theme" as Room36Theme

Item {
    id: root
    objectName: "libraryRoutePlaceholder"

    property string routeId: "library"
    property string title: "Library"
    property string message: ""

    Room36Theme.Theme { id: theme }
    Room36Theme.Typography { id: typography }
    Room36Theme.Metrics { id: metrics }

    ColumnLayout {
        anchors.centerIn: parent
        width: Math.max(0, parent.width - metrics.spacingLarge * 2)
        spacing: metrics.spacingMedium

        Text {
            Layout.fillWidth: true
            text: root.title
            color: theme.mainBlue
            font.family: typography.pageTitleFamily
            font.pixelSize: typography.pageTitle
            wrapMode: Text.Wrap
            horizontalAlignment: Text.AlignHCenter
        }

        Text {
            Layout.fillWidth: true
            text: root.message
            color: theme.mutedText
            font.family: typography.bodyFamily
            font.pixelSize: typography.body
            wrapMode: Text.Wrap
            horizontalAlignment: Text.AlignHCenter
        }

        Text {
            Layout.fillWidth: true
            text: "Phase 4 navigation preview"
            color: theme.mutedText
            font.family: typography.captionFamily
            font.pixelSize: typography.caption
            horizontalAlignment: Text.AlignHCenter
        }
    }
}
