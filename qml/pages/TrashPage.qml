pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components" as Components
import "../theme" as Room36Theme

Item {
    id: root
    objectName: "trashPage"
    property var trash: null
    property var favorites: null
    property var inspector: null
    property string viewSize: "Medium"
    property var confirmPlan: ({})
    Room36Theme.Theme { id: theme }
    Room36Theme.Typography { id: typography }
    Component.onDestruction: if (inspector) inspector.close()

    ColumnLayout {
        anchors.fill: parent
        spacing: 10
        Text {
            Layout.fillWidth: true
            text: "Review photos marked for removal before moving them to the operating-system Trash."
            color: theme.mainBlue
            wrapMode: Text.WordWrap
            font.pixelSize: typography.body
            Accessible.name: text
        }
        RowLayout {
            Layout.fillWidth: true
            Components.ToolbarDropdownButton {
                objectName: "trashFilter"
                label: root.trash ? root.trash.photoModel.candidateFilter : "All"
                options: ["All", "Ready to remove", "Needs attention"]
                onOptionSelected: function(value) { if (root.trash) root.trash.photoModel.setFilter(value) }
                Accessible.name: "Filter pending removal photos: " + label
            }
            Components.PillButton {
                objectName: "trashSelectAllReady"
                variant: "translucent"
                text: "Select all ready"
                enabled: root.trash !== null && root.trash.projectionReady && root.trash.readyCount > 0
                    && root.trash.photoModel.candidateFilter !== "Needs attention"
                onClicked: root.trash.photoModel.selectAllReady()
                Accessible.name: "Select all photos ready to move to Trash"
            }
            Components.PillButton {
                objectName: "trashClearSelection"
                variant: "translucent"
                text: "Clear selection"
                enabled: root.trash !== null && root.trash.photoModel.selectedCount > 0 && !root.trash.busy
                onClicked: root.trash.photoModel.clearSelection()
                Accessible.name: "Clear Trash batch selection"
            }
            Item { Layout.fillWidth: true }
            Components.PillButton {
                objectName: "trashMoveSelected"
                variant: "accent"
                text: "Move selected to Trash"
                enabled: root.trash !== null && root.trash.projectionReady
                    && root.trash.photoModel.selectedCount > 0
                onClicked: {
                    root.confirmPlan = root.trash.prepareSelection()
                    if (root.confirmPlan.count) trashConfirmation.open()
                }
                Accessible.name: "Review selected photos before moving them to Trash"
            }
        }
        Text {
            objectName: "trashStateMessage"
            Layout.fillWidth: true
            visible: root.trash === null || !root.trash.loaded || root.trash.count === 0 || root.trash.error.length > 0
                || root.trash.summary.length > 0 || root.trash.attentionCount > 0
            text: !root.trash ? "Loading removal decisions…"
                : root.trash.error.length > 0 && root.trash.summary.length > 0
                    ? root.trash.summary + "\n" + root.trash.error
                : root.trash.error.length > 0 ? root.trash.error
                : !root.trash.loaded ? "Loading removal decisions…"
                : root.trash.summary.length > 0 ? root.trash.summary
                : root.trash.count === 0 ? "No photos are marked for removal."
                : root.trash.readyCount === 0 ? root.trash.attentionCount + " photos need attention before they can be moved to Trash."
                : root.trash.attentionCount > 0 ? root.trash.attentionCount + " photos need attention before removal." : ""
            color: theme.mainBlue
            wrapMode: Text.WordWrap
            font.pixelSize: typography.body
            Accessible.name: text
        }
        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 12
            Components.PhotoGrid {
                objectName: "trashGrid"
                Layout.fillWidth: true
                Layout.fillHeight: true
                photoModel: root.trash ? root.trash.photoModel : null
                enabled: root.trash !== null && !root.trash.busy
                selectionController: root.trash
                favoritesController: root.favorites
                viewSize: root.viewSize
                openOnClick: true
                emptyText: root.trash && root.trash.error.length > 0 ? "Removal decisions are unavailable."
                    : !root.trash || !root.trash.loaded ? "Loading removal decisions…"
                    : root.trash.count === 0 ? "No photos are marked for removal."
                    : "No pending photos match this filter."
                onPhotoActivated: function(identity) {
                    if (root.inspector && root.trash)
                        root.inspector.openFromModel(root.trash.photoModel, identity)
                }
            }
            ListView {
                id: decisionList
                objectName: "trashDecisionList"
                Layout.preferredWidth: Math.min(300, Math.max(230, root.width * 0.30))
                Layout.fillHeight: true
                clip: true
                spacing: 8
                model: root.trash ? root.trash.photoModel : null
                ScrollBar.vertical: ScrollBar {}
                delegate: Rectangle {
                    id: decisionRow
                    required property string photoId
                    required property string filename
                    required property var removalSources
                    required property string conflictReason
                    required property bool executable
                    required property bool selected
                    width: decisionList.width - 14
                    height: Math.max(118, decisionText.implicitHeight + 78)
                    radius: 14
                    color: theme.panelFill
                    border.width: 1
                    border.color: theme.separator
                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 10
                        spacing: 4
                        Text {
                            Layout.fillWidth: true
                            text: decisionRow.filename
                            color: theme.mainBlue
                            font.pixelSize: typography.body
                            elide: Text.ElideMiddle
                        }
                        Text {
                            id: decisionText
                            Layout.fillWidth: true
                            text: decisionRow.conflictReason.length > 0
                                ? decisionRow.conflictReason
                                : "Marked for removal: " + decisionRow.removalSources.join(" · ")
                            color: theme.mutedText
                            font.pixelSize: typography.caption
                            wrapMode: Text.WordWrap
                            Accessible.name: text
                        }
                        RowLayout {
                            Components.SelectionCheckbox {
                                objectName: "trashSelectionCheckbox"
                                checked: decisionRow.selected
                                enabled: decisionRow.executable && root.trash !== null && !root.trash.busy
                                onToggled: if (root.trash) root.trash.photoModel.setSelected(decisionRow.photoId, checked)
                                Accessible.name: "Select " + decisionRow.filename + " for this Trash batch"
                            }
                            Item { Layout.fillWidth: true }
                            Components.PillButton {
                                objectName: "trashRestorePhoto"
                                variant: "translucent"
                                text: "Cancel removal"
                                enabled: root.trash !== null && root.trash.projectionReady
                                onClicked: root.trash.restorePhoto(decisionRow.photoId)
                                Accessible.name: "Cancel all removal decisions for " + decisionRow.filename
                            }
                        }
                    }
                    Accessible.name: decisionRow.filename + ", " + decisionText.text
                }
            }
        }
    }

    Dialog {
        id: trashConfirmation
        objectName: "trashConfirmation"
        parent: Overlay.overlay
        anchors.centerIn: parent
        modal: true
        title: "Move photos to Trash?"
        width: Math.min(460, root.width - 40)
        standardButtons: Dialog.NoButton
        background: Rectangle {
            radius: 18
            color: theme.panelFill
            border.width: 1
            border.color: theme.separator
        }
        ColumnLayout {
            width: trashConfirmation.availableWidth
            spacing: 12
            Text {
                Layout.fillWidth: true
                text: "Move " + (root.confirmPlan.count || 0) + " photos (" +
                    Math.round((root.confirmPlan.totalBytes || 0) / 1048576) +
                    " MB) to the operating-system Trash?"
                color: theme.mainBlue
                wrapMode: Text.WordWrap
                Accessible.name: text
            }
            Text {
                Layout.fillWidth: true
                text: "Blur: " + (root.confirmPlan.blur || 0) + " · Near duplicates: " +
                    (root.confirmPlan.duplicates || 0) + " · Both: " + (root.confirmPlan.both || 0)
                color: theme.mutedText
                Accessible.name: text
            }
            RowLayout {
                Item { Layout.fillWidth: true }
                Components.PillButton {
                    objectName: "trashCancelConfirmation"
                    variant: "translucent"
                    text: "Cancel"
                    onClicked: trashConfirmation.close()
                    Accessible.name: "Cancel moving photos to Trash"
                }
                Components.PillButton {
                    objectName: "trashConfirmMove"
                    variant: "accent"
                    text: "Move to Trash"
                    onClicked: {
                        if (root.trash) root.trash.executeSelection()
                        trashConfirmation.close()
                    }
                    Accessible.name: "Confirm moving selected photos to the operating-system Trash"
                }
            }
        }
    }
    Components.PhotoInspector {
        anchors.fill: parent
        controller: root.inspector
        favoritesController: root.favorites
        showReviewActions: false
        visible: controller !== null && controller.opened
    }
}
