pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../theme" as Room36Theme

Item {
    id: root
    objectName: "peopleDetail"
    property var person: ({})
    property var photoModel: null
    property var inspector: null
    property var favoritesController: null
    property var exporter: null
    property bool allowPersonExport: false
    property string viewSize: "Medium"
    property string title: ""
    property Component actions: null
    signal backRequested()
    Room36Theme.Theme { id: theme }
    Room36Theme.Typography { id: typography }
    Component.onDestruction: if (inspector) inspector.close()

    ColumnLayout {
        anchors.fill: parent
        spacing: 12
        RowLayout {
            Layout.fillWidth: true
            spacing: 16
            PillButton {
                objectName: "peopleDetailBack"
                variant: "translucent"
                text: "← Back"
                onClicked: root.backRequested()
                Accessible.name: "Back to people"
            }
            Rectangle {
                Layout.preferredWidth: 70
                Layout.preferredHeight: 70
                radius: 24
                color: theme.imagePlaceholder
                clip: true
                Image {
                    anchors.fill: parent
                    source: root.person.previewSource || ""
                    asynchronous: true
                    cache: false
                    fillMode: Image.PreserveAspectCrop
                    Accessible.name: root.title + " face preview"
                }
            }
            ColumnLayout {
                Layout.fillWidth: true
                Text {
                    Layout.fillWidth: true
                    text: root.title
                    color: theme.mainBlue
                    font.pixelSize: 24
                    elide: Text.ElideRight
                }
                Text {
                    text: (root.person.photoCount || 0) + " photos · " + (root.person.faceCount || 0) + " faces"
                    color: theme.mutedText
                    font.pixelSize: typography.body
                }
            }
            PillButton {
                objectName: "peopleExportSelected"
                variant: "translucent"
                text: "Export selected"
                enabled: root.exporter !== null && !root.exporter.busy && !root.exporter.preparing
                    && root.photoModel !== null && root.photoModel.selectedCount > 0
                onClicked: root.exporter.prepareSelected(root.photoModel)
                Accessible.name: "Export selected person photos"
            }
            PillButton {
                objectName: "knownExportPerson"
                visible: root.allowPersonExport
                variant: "accent"
                text: "Export person"
                enabled: root.exporter !== null && !root.exporter.busy && !root.exporter.preparing
                onClicked: root.exporter.preparePerson()
                Accessible.name: "Export all photos of this known person"
            }
            Loader { sourceComponent: root.actions }
        }
        PhotoGrid {
            Layout.fillWidth: true
            Layout.fillHeight: true
            photoModel: root.photoModel
            favoritesController: root.favoritesController
            viewSize: root.viewSize
            openOnClick: true
            emptyText: "No project photos are available for this person."
            onPhotoActivated: function(identity) {
                if (root.inspector) root.inspector.openFromModel(root.photoModel, identity)
            }
        }
    }
    Component {
        id: inspectorDetails
        ColumnLayout {
            Text { text: root.title; color: theme.mainBlue; font.pixelSize: 20 }
            Text { text: "Project photo"; color: theme.mutedText }
        }
    }
    PhotoInspector {
        anchors.fill: parent
        controller: root.inspector
        favoritesController: root.favoritesController
        detailsComponent: inspectorDetails
        showReviewActions: false
        visible: controller !== null && controller.opened
    }
}
