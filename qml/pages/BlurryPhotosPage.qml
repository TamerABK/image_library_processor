pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components" as Components
import "../theme" as Room36Theme

Item {
    id: root
    objectName: "blurryPhotosPage"
    property var processing: null
    property var inspector: null
    property var favorites: null
    property var exporter: null
    property string viewSize: "Medium"
    readonly property var blurStage: processing && processing.stages.length > 0
        ? processing.stages[0] : ({state: "NOT_STARTED", error: ""})
    Room36Theme.Theme { id: theme }
    Component.onDestruction: if (inspector) inspector.close()

    ColumnLayout {
        anchors.fill: parent
        spacing: 12
        RowLayout {
            Layout.fillWidth: true
            Components.ToolbarDropdownButton {
                objectName: "blurSort"
                implicitWidth: 210
                normalFill: theme.separator
                label: root.processing ? root.processing.blurModel.sortOrder : "Most problematic first"
                readonly property string currentText: label
                options: ["Most problematic first", "Least problematic first", "Filename A–Z", "Filename Z–A"]
                onOptionSelected: function(option) {
                    if (root.processing) root.processing.blurModel.setSort(option)
                }
                Accessible.name: "Sort blur candidates: " + label
            }
            Components.ToolbarDropdownButton {
                objectName: "blurFilter"
                implicitWidth: 170
                normalFill: theme.separator
                label: root.processing ? root.processing.blurModel.classificationFilter : "All candidates"
                readonly property string currentText: label
                options: ["All candidates", "Blurry", "Review"]
                onOptionSelected: function(option) {
                    if (root.processing) root.processing.blurModel.setFilter(option)
                }
                Accessible.name: "Filter blur candidates: " + label
            }
            Item { Layout.fillWidth: true }
            Components.PillButton {
                objectName: "blurExportSelected"
                variant: "translucent"
                text: "Export selected"
                enabled: root.exporter !== null && !root.exporter.busy && !root.exporter.preparing
                    && root.processing !== null && root.processing.blurModel.selectedCount > 0
                onClicked: root.exporter.prepareSelected(root.processing.blurModel)
                Accessible.name: "Export selected blurry photos"
            }
            Components.PrimaryButton {
                objectName: "bulkKeep"
                text: "Keep"
                implicitWidth: 90
                enabled: root.processing !== null && root.processing.blurModel.selectedCount > 0
                onClicked: root.processing.reviewSelected("KEEP")
                Accessible.name: "Keep selected photos"
            }
            Components.PrimaryButton {
                objectName: "bulkReview"
                text: "Review"
                implicitWidth: 90
                enabled: root.processing !== null && root.processing.blurModel.selectedCount > 0
                onClicked: root.processing.reviewSelected("REVIEW")
                Accessible.name: "Mark selected photos for review"
            }
            Components.PrimaryButton {
                objectName: "bulkRemove"
                text: "Remove"
                implicitWidth: 90
                enabled: root.processing !== null && root.processing.blurModel.selectedCount > 0
                onClicked: root.processing.reviewSelected("REMOVE")
                Accessible.name: "Mark selected photos for future removal"
            }
        }
        Text {
            Layout.fillWidth: true
            visible: root.blurStage.error.length > 0 || (root.processing !== null && root.processing.message.length > 0)
            text: root.processing && root.processing.message.length > 0 ? root.processing.message : root.blurStage.error
            color: theme.mainBlue
            wrapMode: Text.WordWrap
        }
        Components.PhotoGrid {
            id: grid
            Layout.fillWidth: true
            Layout.fillHeight: true
            photoModel: root.processing ? root.processing.blurModel : null
            favoritesController: root.favorites
            viewSize: root.viewSize
            openOnClick: true
            selectOnImageClick: true
            emptyText: root.blurStage.state === "FAILED" ? "Image quality analysis could not finish."
                : root.blurStage.state === "COMPLETE" ? "No blurry photos need review."
                : "Analyzing image quality…"
            onPhotoActivated: function(identity) { if (root.inspector) root.inspector.open(identity) }
        }
    }
    Components.PhotoInspector {
        anchors.fill: parent
        controller: root.inspector
        favoritesController: root.favorites
        visible: controller !== null && controller.opened
        onReviewRequested: function(identity, intent) { if (root.processing) root.processing.setReview(identity, intent) }
    }
}
